"""
模型训练、超参数调优、评估 和 SHAP
"""
import os
import warnings
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from catboost import CatBoostClassifier , Pool

from .config import (
    CATBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    OPTUNA_TRIALS,
    OPTUNA_TIMEOUT,
    OUTPUT_DIR,
    MODEL_PATH,
    RANDOM_SEED,

       )

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False




# 数据拆分

def split_by_time(
        df: pd.DataFrame,
        split_ratio: float = 0.7,
        time_col: str = "hours_since_start",
) -> tuple:
    
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        print(f'{time_col} not found in columns')
    
    split_idx = int(len(df) * split_ratio)

    train_data = df[:split_idx]
    valid_data = df[split_idx:]

    y_train = train_data['label'].reset_index(drop=True)
    y_valid = valid_data['label'].reset_index(drop=True)

    X_train = train_data.drop(columns = ['label']).reset_index(drop=True)
    X_valid = valid_data.drop(columns = ['label']).reset_index(drop=True)

    print(f" Train : {X_train.shape}，Valid : {X_valid.shape}")
    print(f" Train cvr : {y_train.mean()} ,Valid cvr : {y_valid.mean()}")

    # 生成训练集的mask码
    train_mask = np.zeros(len(df),dtype=bool)
    train_mask[:split_idx] = True

    return X_train, X_valid, y_train, y_valid, train_mask


# catboost 
def train_catboost(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        cat_features: list[str],
        params:dict | None = None,
        use_pool: bool = True,

)->CatBoostClassifier:
    """ 
    训练CatBoost
    """
    if params is None:
        params = CATBOOST_PARAMS.copy()
    
    # 筛选cat_features中在X_train中的特征
    cat_feats_present = [c for c in cat_features if c in X_train.columns]
    print(f"{len(cat_feats_present)} categorical features")

    # 确保是字符串类型
    for col in cat_feats_present:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_valid[col] = X_valid[col].astype(str)
    
    if use_pool:
        train_pool = Pool(X_train, y_train, cat_features=cat_feats_present)
        valid_pool = Pool(X_valid, y_valid, cat_features=cat_feats_present)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set = valid_pool,
            early_stopping_rounds = params.get("early_stopping_rounds", 50),
            verbose = params.get("verbose", 100),
        )
    else:
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set = (X_valid, y_valid),
            cat_features = cat_feats_present,
            early_stopping_rounds = params.get("early_stopping_rounds", 50),
            verbose = params.get("verbose", 100),
            plot=False,
        )
    
    best_score = model.get_best_score()
    print(f" best auc: {best_score['validation']['AUC']:.4f}"
          f"(iter {model.get_best_iteration()})")

    return model


# lightgbm
def train_lightgbm(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        cat_features: list[str],
        params:dict | None = None,
):
    """
    训练LightGBM
    """
    import lightgbm as lgb

    if params is None:
        params = LIGHTGBM_PARAMS.copy()
    
    cat_feats_present = [c for c in cat_features if c in X_train.columns]

    # lightgbm 需 categorical 特征转 category
    for col in cat_feats_present:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")

    train_data = lgb.Dataset(
        X_train,
        label =  y_train,
        categorical_feature = cat_feats_present,
    )
    valid_data = lgb.Dataset(
        X_valid, label= y_valid, 
        categorical_feature = cat_feats_present,
        reference = train_data,
    )
    
    model = lgb.train(
        params,
        train_data,
        valid_sets = [train_data,valid_data],
        valid_names = ['train','valid'],
        num_boost_round = 1000,
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    best_auc = model.best_score['valid']['auc']
    print(f'best auc: {best_auc:.4f} (iter {model.best_iteration})')

    return model

# 超参数调优
def optimize_catboost_optuna(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        cat_features: list[str],
        n_trials: int = OPTUNA_TRIALS,
        timeout:int = OPTUNA_TIMEOUT,
)-> dict:
    """ 
    超参数调优
    """

    try: 
        import optuna
    except ImportError:
        print("Optuna is not installed. Please install it first.")
        return CATBOOST_PARAMS.copy()
    
    
    # 用30%随机无重复的训练数据，代替全量数据做optuna调优
    sample_frac = 0.30
    print(f"{sample_frac:.0%} of training data for search speed...")
    
    np.random.seed(RANDOM_SEED)
    n_sample = int(len(X_train) * sample_frac)
    sample_idx = np.random.choice(len(X_train), size=n_sample, replace=False)
    

    X_sample = X_train.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_train.iloc[sample_idx].reset_index(drop=True)

    # 同样对验证集进行采样
    val_n = min(len(X_valid), 200000)
    val_idx  = np.random.choice(len(X_valid),size = val_n, replace = False)
    X_val_sample = X_valid.iloc[val_idx].reset_index(drop = True)
    y_val_sample = y_valid.iloc[val_idx].reset_index(drop = True)

    print(f"搜索的数据：train = {n_sample} ,valid = {val_n}")

    cat_feats_present = [c for c in cat_features if c in X_sample.columns]
    for col  in cat_feats_present:
        X_sample[col] = X_sample[col].astype(str)
        X_val_sample[col] = X_val_sample[col].astype(str)
    
    train_pool = Pool(X_sample, y_sample, cat_features=cat_feats_present)
    valid_pool = Pool(X_val_sample, y_val_sample, cat_features=cat_feats_present)

    # 统计次数展示
    trial_count = [0]
    
    def objective(trial):
        params = {
            "iterations":trial.suggest_int("iterations", 200, 600, step = 100),
            "learning_rate":trial.suggest_float("learning_rate",0.01,0.1,log=True),
            "depth":trial.suggest_int("depth",4,10),
            "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1,10,log=True),
            "subsample":trial.suggest_float("subsample",0.7,1.0),
            "colsample_bylevel":trial.suggest_float("colsample_bylevel",0.7,1.0),
            "loss_function":"Logloss",
            "eval_metric":"AUC",
            "random_seed":RANDOM_SEED,
            "verbose":0,
            "early_stopping_rounds":30,
            "auto_class_weights":"Balanced",
            "thread_count": -1,
        }
        trial_count[0] += 1
        print(f"  Trial {trial_count[0]}/{n_trials} (lr={trial.params.get('learning_rate', '?'):.3f}, "
              f"depth={trial.params.get('depth', '?')})...", end=" ", flush=True)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set = valid_pool,
            early_stopping_rounds = 30,
            verbose = 0,
        )

        auc = model.get_best_score()['validation']["AUC"]
        print(f"  AUC: {auc:.4f}")
        return auc
    
    
    # 关了optuna自带的进度条
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        pruner = optuna.pruners.MedianPruner(n_startup_trials = 3)

    )

    study.optimize(
        objective,
        n_trials = n_trials,
        timeout = timeout,
        show_progress_bar = True,
    )

    print(f"\n  Best trial (#{study.best_trial.number}):")
    print(f"  AUC: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    # 合并最佳参数与基础参数
    best_params = CATBOOST_PARAMS.copy()
    best_params.update(study.best_params)
    best_params["verbose"] = 100

    return best_params




# 评估

def evaluate_model(
        model,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        cat_features: list[str] | None = None,
        model_name: str = "CatBoost",

)-> dict:
    """ 
    评估模型
    """

    # 根据各类模型，将分类数据转化为对应的格式
    is_lgb= not hasattr(model, "predict_proba")
    if cat_features:
        for col in cat_features:
            if col in X_valid.columns:
                if is_lgb:
                    X_valid[col] = X_valid[col].astype("category")
                else:
                    X_valid[col] = X_valid[col].astype(str)
    
    # 处理两个模型的predict_proba接口问题
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_valid)[:, 1]
    else:
        # LightGBM Booster: predict () 函数在二分类任务中返回各类别概率值
        y_proba = model.predict(X_valid, num_iteration=model.best_iteration)
        # 一维
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        else:
            y_proba = y_proba.ravel()


    # auc
    auc = roc_auc_score(y_valid, y_proba)

    # pr_auc
    pr_auc = average_precision_score(y_valid, y_proba)

    # best threshold via f1
    precision, recall,thresholds = precision_recall_curve(y_valid,y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10) # 极小值防报错
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx] if len(thresholds) > best_thresh_idx else 0.5

    y_pred = (y_proba >= best_threshold).astype(int)

    # f1
    f1 = f1_score(y_valid, y_pred)
    prec = precision_score(y_valid, y_pred)
    rec = recall_score(y_valid, y_pred)


    # confusion matrix
    cm = confusion_matrix(y_valid, y_pred)


    # 输出一下
    print(f"  AUC: {auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  Confusion Matrix:")
    print(f"  TN: {cm[0,0]:,}, FP = {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}, TP = {cm[1,1]:,}")


    # 可视化 roc + pr  curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # roc
    fpr, tpr, _ = roc_curve(y_valid, y_proba)
    axes[0].plot(fpr, tpr, label=f"{model_name} AUC = {auc:.4f}",lw=2)
    axes[0].plot([0,1],[0,1],"k--",alpha = 0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title("Roc Curve")
    axes[0].legend()
    axes[0].grid(True,alpha=0.3)


    # PR
    axes[1].plot(recall, precision, label=f"{model_name} (PR-AUC={pr_auc:.4f})", lw=2, color="orange")
    axes[1].axhline(y=y_valid.mean(), color="k", linestyle="--", alpha=0.3,
                    label=f"Random ({y_valid.mean():.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)




    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved to: {plot_path}")

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "best_threshold": best_threshold,
        "confusion_matrix": cm,
    }


# 特征重要性

def analyze_feature_importance(
        model,
        feature_names:list[str],
        top_n: int = 30,
        model_name : str = "CatBoost",

)-> pd.DataFrame:
    """
    分析特征重要性
    """

    print(f"  Feature Importances:{model_name}")

    # 自适应处理不同模型的特征重要性
    if hasattr(model, "get_feature_importance"):
        importance = model.get_feature_importance()
    elif hasattr(model, "feature_importance"):
        importance = model.feature_importance(importance_type="gain")
    else:
        print("  Warning: cannot extract feature importance from this model")
        return pd.DataFrame(columns=["feature", "importance"])


    # 特征重要性对齐
    importance = np.array(importance)
    if len(importance) != len(feature_names):
        print(f"  Warning: importance length ({len(importance)}) != feature_names ({len(feature_names)}), using available")
        feature_names = feature_names[:len(importance)]

    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    # 保存一下
    imp_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_feature_importance.csv")
    feat_imp_df.to_csv(imp_path, index=False)
    print(f"  Saved to {imp_path}")


    # Print top N
    print(f"\n  Top {top_n} features:")
    print(feat_imp_df.head(top_n).to_string(index=False))

 
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    top_features = feat_imp_df.head(top_n).iloc[::-1]
    ax.barh(range(len(top_features)), top_features["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance ({model_name})")
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_feature_importance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {plot_path}")

    return feat_imp_df


# SHAP
def shap_analysis(
    model,
    X_sample: pd.DataFrame,
    cat_features: list[str],
    model_name: str = "CatBoost",
    sample_size: int = 5000,
):
    """
    分析模型的SHAP解释性
    """


    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Skipping.")
        return

    # 确保分类特征是字符串类型
    for col in cat_features:
        if col in X_sample.columns:
            X_sample[col] = X_sample[col].astype(str)

    # 采样数据
    X_shap = X_sample.sample(
    n=min(len(X_sample), sample_size),  # 自动取较小值，无需if判断
    random_state=RANDOM_SEED,
    replace=False  # 明确禁止有放回抽样，更规范
    )


    print(f"  Computing SHAP values on {len(X_shap)} samples...")

    # 调用catboost自带的方法计算shap值
    shap_values = model.get_feature_importance(
        data=Pool(X_shap, cat_features=[c for c in cat_features if c in X_shap.columns]),
        type="ShapValues",
    )

    # 移除最后一行偏置项
    shap_values_features = shap_values[:, :-1]

    # 绘制SHAP摘要性图表
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values_features,
        X_shap,
        feature_names=list(X_shap.columns),
        max_display=20,
        show=False,
    )
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_shap_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary plot saved to {plot_path}")

    # 基于SHAP值的特征重要性
    shap_importance = np.abs(shap_values_features).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "feature": list(X_shap.columns),
        "shap_importance": shap_importance,
    }).sort_values("shap_importance", ascending=False)

    shap_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_shap_importance.csv")
    shap_imp_df.to_csv(shap_path, index=False)
    print(f"  SHAP importance saved to {shap_path}")

    print("\n  Top 10 SHAP features:")
    print(shap_imp_df.head(10).to_string(index=False))


# 模型保存 预测
def save_model(model, path: str = MODEL_PATH):
    """将训练好的模型保存至磁盘"""
    model.save_model(path)
    print(f"\n  Model saved to: {path}")


def predict_test(
    model,
    X_test: pd.DataFrame,
    cat_features: list[str],
    threshold: float = 0.5,
) -> np.ndarray:
    """
    生成测试集的预测概率
    """

    cat_feats_present = [c for c in cat_features if c in X_test.columns]
    for col in cat_feats_present:
        X_test[col] = X_test[col].astype(str)

    test_pool = Pool(X_test, cat_features=cat_feats_present)
    y_proba = model.predict_proba(test_pool)[:, 1]
    print(f"  Predictions generated: {len(y_proba)} samples")
    print(f"  Predicted positive rate: {(y_proba >= threshold).mean():.4f}")

    return y_proba

