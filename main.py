#!/usr/bin/env python3

""" 
梳理下流程：
1. 数据加载和清洗
2. 特征工程
3. 模型训练catboost
4. optuna超参数优化
5. 对比模型lightgbm
6. 模型评估
7. shap分析

Usage:
    python main.py                    # Run full pipeline (CatBoost + LightGBM)
    python main.py --optuna           # Also run Optuna hyperparameter search
    python main.py --shap             # Also run SHAP interpretability
    python main.py --predict          # Also generate test set predictions  不建议跑predict预测无标签的test，特征缺失较多，越改越乱。
    python main.py --skip-lightgbm    # Skip LightGBM comparison (faster)
    
    我已将优化的超参数保存到config.py中作为模型更新参数，若需重跑optuna，需将config.py优化的超参数注释，解开原始参数
            D:\anaconda\envs\data_ana_project_env\python main.py --optuna
      Params: {'iterations': 600, 'learning_rate': 0.03649155981381801, 
      'depth': 10, 'l2_leaf_reg': 7.886332236196301, 'subsample': 0.9982548767248413, 'colsample_bylevel': 0.8755462322917292}

 
"""


import argparse
import time
import numpy as np
import pandas   as pd
import warnings

from src.config import (
    SPLIT_RATIO,
    CATBOOST_PARAMS,
    RANDOM_SEED,
    OUTPUT_DIR,
    SUBMISSION_PATH,
)
from src.data import load_and_prepare_data
from src.features import prepare_features, prepare_test_features
from src.models import (
    split_by_time,
    train_catboost,
    train_lightgbm,
    optimize_catboost_optuna,
    evaluate_model,
    analyze_feature_importance,
    shap_analysis,
    save_model,
    predict_test,
)
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


def main():
    parser = argparse.ArgumentParser(
        description = "Ad Conversion Rate Prediction - Pipeline"
    )
    parser.add_argument(
        '--optuna', action = 'store_true',
        help = 'Enable Optuna hyperparameter optimization'

    )
    parser.add_argument(
        "--shap",action = "store_true",
        help = "Enable Shap analysis"
    )
    parser.add_argument(
        "--skip-lightgbm", action = "store_true",
        help = 'Skip LightGBM comparison'
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Enable prediction on test set"
    )
    # 解析parse_args传入args
    args = parser.parse_args()

    start_time = time.time()

    # 1. load data - clean - merge auxiiliary tables - 
    # extract user behavior features - optimize  memory

    result = load_and_prepare_data(use_test=args.predict)
    df_full = result["train_val"]
    test_df = result.get("test")


    # 2. 确保时序
    if "hours_since_start" not in df_full.columns:
        click_time_str = df_full["clickTime"].astype(str).str.zfill(6)
        df_full["hours_since_start"] = (
            click_time_str.str[0:2].astype(int) * 24
            + click_time_str.str[2:4].astype(int)
            + click_time_str.str[4:6].astype(int) / 60.0
        )

    # 拆分，只拿train_mask防止target encoding泄露
    _, _, y_train_raw, y_valid_raw ,train_mask = split_by_time(
        df_full,split_ratio = SPLIT_RATIO,  time_col = "hours_since_start"
    )

    # 3. 特征工程

    # 完整的特征矩阵
    df_featurized, cat_features, num_features = prepare_features(
        df_full,
        y=df_full["label"],
        train_mask=train_mask,
        fit_encodings=True,
    )

    # 拆分训练集和验证集
    split_idx = int(len(df_featurized) * SPLIT_RATIO)
    df_featurized = df_featurized.sort_values("hours_since_start").reset_index(drop=True)

    train_df = df_featurized.iloc[:split_idx]
    valid_df = df_featurized.iloc[split_idx:]   


    y_train = train_df['label'].reset_index(drop=True)
    y_valid = valid_df['label'].reset_index(drop=True)
    X_train = train_df.drop(columns=['label']).reset_index(drop=True)
    X_valid = valid_df.drop(columns=['label']).reset_index(drop=True)

    print(f"final train: {X_train.shape}  ,valid: {X_valid.shape}")


    # 4. 模型训练catboost
    # --- Hyperparameter optimization (optional, opt-in) ---
    if args.optuna:
        best_params = optimize_catboost_optuna(
            X_train, y_train, X_valid, y_valid, cat_features
        )
    else:
        print("\n  Skipping Optuna (use --optuna to enable)")
        best_params = CATBOOST_PARAMS.copy()

    # --- Train CatBoost with best params ---
    catboost_model = train_catboost(
        X_train, y_train, X_valid, y_valid, cat_features, params=best_params
    )

    # Save model
    save_model(catboost_model)

    # 5. 评估
    eval_metrics = evaluate_model(
        catboost_model, X_valid, y_valid, cat_features, model_name="CatBoost"
    )

    # Feature importance
    feat_imp = analyze_feature_importance(
        catboost_model, list(X_train.columns), model_name="CatBoost"
    )


    # 6. 对比模型lightgbm
    if not args.skip_lightgbm:
        lgb_model = train_lightgbm(
            X_train, y_train, X_valid, y_valid, cat_features
        )
        lgb_eval = evaluate_model(
            lgb_model, X_valid, y_valid, cat_features, model_name="LightGBM"
        )
        lgb_feat_imp = analyze_feature_importance(
            lgb_model, list(X_train.columns), model_name="LightGBM"
        )

        # Compare
        print(f"  CatBoost AUC: {eval_metrics['auc']:.4f}, "
                f"F1: {eval_metrics['f1']:.4f}, "
                f"PR-AUC: {eval_metrics['pr_auc']:.4f}")
        print(f"  LightGBM AUC: {lgb_eval['auc']:.4f}, "
                f"F1: {lgb_eval['f1']:.4f}, "
                f"PR-AUC: {lgb_eval['pr_auc']:.4f}")
        
    # 7. SHAP分析
    if args.shap:
        shap_analysis(catboost_model, X_valid, cat_features, model_name="CatBoost")


    # 8. 测试集预测
    if args.predict and test_df is not None:
        X_test = prepare_test_features(test_df, cat_features, num_features)
        y_test_proba = predict_test(
            catboost_model, X_test, cat_features, threshold=eval_metrics["best_threshold"]
        )

        # Save submission
        submission = pd.DataFrame({
            "id": test_df["userID"].values,
            "prediction": y_test_proba,
        })

        # If userID is not unique, handle accordingly
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"  Saved to {SUBMISSION_PATH}")


    # 9. 总结
    elapsed = time.time() - start_time 
    print(f"  Total time:     {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  CatBoost AUC:   {eval_metrics['auc']:.4f}")
    print(f"  CatBoost F1:    {eval_metrics['f1']:.4f}")
    print(f"  CatBoost PR-AUC: {eval_metrics['pr_auc']:.4f}")
    print(f"  Output dir:     {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
