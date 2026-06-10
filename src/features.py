"""
梳理一下：
时序特征：循环编码 ，时间分桶
地域/应用 类目拆分
目标编码
高基数特征计数编码
交叉特征
特征筛选
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .config import (
    HIGH_CARDINALITY_FEATURES,
    CROSS_FEATURE_PAIRS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    MAX_CAT_COUNT,
)

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从clickTime提取时间特征
    """
    
    click_time_str = df['clickTime'].astype(str).str.zfill(6)

    df['day'] = click_time_str.str[0:2].astype(int)
    df['hour'] = click_time_str.str[2:4].astype(int)
    df['minute'] = click_time_str.str[4:6].astype(int)

    # 构造连续时间
    df['hours_since_start'] = (
    df['day'] * 24 + df['hour'] + df['minute'] / 60.0
    )

    # 把hour作为category
    df['click_hour'] = df['hour'].astype('int8').astype(str)

    # hour循环编码
    hour_rad = 2 * np.pi * df['hour'] / 24
    df['click_hour_sin'] = np.sin(hour_rad).astype(np.float32)
    df['click_hour_cos'] = np.cos(hour_rad).astype(np.float32)

    # hour分桶
    def hour_to_bucket(h:int)->str:
        if 0 <= h < 6:
            return "night"
        elif 6 <= h < 12:
            return "morning"
        elif 12 <= h < 18:
            return "afternoon"
        else:
            return "evening"
    
    df['time_of_day'] = df['hour'].apply(hour_to_bucket)

    # Day feature (保留供分析用，后续 exclude_cols 会排除)
    df['click_day'] = df['day'].astype("int8")

    # 删无用列
    df = df.drop(columns = ['day','hour','minute'])

    return df

def decompose_location_and_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    地域/应用 类目拆分
    """
    # hometown
    if "hometown" in df.columns:
        hometown_str = df["hometown"].astype(str).str.zfill(6)
        df["hometown_province"] = hometown_str.str[:2].astype(int).astype("int16")
        df["hometown_city"] = hometown_str.str[2:].astype(int).astype("int16")
        df = df.drop(columns=["hometown"])
        print("  Decomposed hometown -> hometown_province + hometown_city")

    # residence
    if "residence" in df.columns:
        residence_str = df["residence"].astype(str).str.zfill(6)
        df["residence_province"] = residence_str.str[:2].astype(int).astype("int16")
        df["residence_city"] = residence_str.str[2:].astype(int).astype("int16")
        df = df.drop(columns=["residence"])
        print("  Decomposed residence -> residence_province + residence_city")

    # appCategory
    if "appCategory" in df.columns:
        cate_str = df["appCategory"].astype(str).str.zfill(2)
        df["first_category"] = cate_str.str[0].astype(int).astype("int8")
        df["second_category"] = cate_str.str[1:].astype(int).astype("int8")
        df = df.drop(columns=["appCategory"])
        print("  Decomposed appCategory -> first_category + second_category")

    return df

def add_count_encoding(df:pd.DataFrame,train_mask:np.ndarray)->pd.DataFrame:
    """
    高基数特征计数编码
    只用训练集统计频次，再映射到训练和测试集
    """
    train_df = df[train_mask].copy()
    
    for col in HIGH_CARDINALITY_FEATURES:
        if col not in df.columns:
            continue

        # 计数编码  value_counts(): 统计每个值的出现次数
        full_map = train_df[col].value_counts().to_dict()

        # top MAX_CAT_COUNT 保留独立编码，其余归入 "other"
        top_values = set(list(full_map.keys())[:MAX_CAT_COUNT])

        def encode_count(val, mapping=full_map, top=top_values):
            if val in top:
                return mapping.get(val, 0)
            else:
                return 0  # 低频值归零，视为不可靠信号

        df[f'{col}_count'] = df[col].map(encode_count).fillna(0).astype(np.float64)


        # 加个log变换
        df[f"{col}_log_count"] = np.log1p(df[f"{col}_count"]).astype(np.float64)

    return df


def add_target_encoding_time_series(
        df:pd.DataFrame, y:pd.Series, n_splits:int = 5
) -> pd.DataFrame:
    """
    时序交叉目标编码
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    features_to_encode = [c for c in HIGH_CARDINALITY_FEATURES if c in df.columns]

    for col in features_to_encode:
        df[f'{col}_te'] = 0.0

    # 使用时间序列折叠计算编码
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        train_fold = df.iloc[train_idx]
        y_fold = y.iloc[train_idx]
        val_fold_idx = df.index[val_idx]

        for col in features_to_encode:
            # 仅使用训练折数据计算每个类别的目标均值
            te_map = (
                pd.concat([train_fold[col],y_fold],axis=1)
                .groupby(col)["label"]
                .mean()
                .to_dict()
            )

            # 应用到验证折
            df.loc[val_fold_idx, f"{col}_te"] = (
                df.loc[val_fold_idx, col].map(te_map).fillna(y.mean())
            )
        
        if fold == 0:
            print(f"  Fold {fold + 1}/{n_splits}...", end="")
        
        elif fold == n_splits - 1:
            print(f" Fold {fold + 1}/{n_splits} done")
        else:
            print(f" {fold + 1}/{n_splits}...", end="")


    # 对有可能未被包含在验证折中的行，用全局均值填充
    for col in features_to_encode:
        df[f'{col}_te'] = df[f'{col}_te'].replace(0,y.mean())
    
    return df


def add_cross_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    低基数特征交叉编码
    """
    for (f1,f2) in CROSS_FEATURE_PAIRS:
        if f1 in df.columns and f2 in df.columns:
            cross_name = f"{f1}_x_{f2}"
            df[cross_name] = (
                df[f1].astype(str) + "_" + df[f2].astype(str)
            )
            nunique = df[cross_name].nunique()
            
    return df


def prepare_features(
        df:pd.DataFrame,
        y:pd.Series | None = None,
        train_mask:np.ndarray | None = None,
        fit_encodings: bool = True,

) -> tuple[pd.DataFrame, list[str], list[str]]:
    """ 

      Args:
        df: Cleaned and merged DataFrame.
        y: Target labels (required if fit_encodings=True).
        train_mask: Boolean mask for training data (required if fit_encodings=True).
        fit_encodings: If True, compute encodings fit on training data only.

        Returns:
        Tuple of (X, cat_features, num_features):
          - X: Feature matrix with all engineered features
          - cat_features: List of categorical feature column names in X
          - num_features: List of numeric feature column names in X
    """

    # 1.时序
    df = engineer_time_features(df)

    # 2. loc/cat 拆分
    df = decompose_location_and_category(df)

    # 3. count encoding
    if fit_encodings and y is not None and train_mask is not None:
        df = add_count_encoding(df, train_mask)

        # 时序目标编码
        sort_col = "hours_since_start" if "hours_since_start" in df.columns else "clickTime"
        sort_idx = df[sort_col].argsort()
        df_sorted = df.iloc[sort_idx].reset_index(drop = True)
        y_sorted = y.iloc[sort_idx].reset_index(drop = True)

        df_sorted = add_target_encoding_time_series(df_sorted, y_sorted)
        df = df_sorted
    else:
        print("\n Skipping target/count encoding")

    
    # 4. 交叉特征
    df = add_cross_features(df)

    # 删除非特征列
    exclude_cols = {
        "label","clickTime",
        "conversionTime","click_day",
        "camgaignID","index"
    }
    
    # 看看实际特征列
    all_cols = set(df.columns) - exclude_cols
  
    # 划分分类特征
    cat_features = []
    num_features = []

    for col in sorted(all_cols):
        if col in df.columns:
            dtype = df[col].dtype
            if dtype in ("object",'category','string'):
                cat_features.append(col)
            elif col in NUMERIC_FEATURES:
                num_features.append(col)
            elif col in CATEGORICAL_FEATURES:
                cat_features.append(col)
            elif col.endswith("_te") or col.endswith("_count") or col.endswith("_log_count"):
                num_features.append(col)
            elif col in ("click_hour_sin","click_hour_cos","hours_since_start",
                        "user_action_count_before","user_unique_apps_before"):
                num_features.append(col)
            elif col.startswith("user_"):
                num_features.append(col)
            elif dtype in ("int8","int16","int32","int64","float32","float64"):
                # 低基数 ints categorical
                n_unique = df[col].nunique()
                if n_unique <= 50 and col not in ("age",):
                    cat_features.append(col)
                else:
                    num_features.append(col)
            else:
                cat_features.append(col)

    # 把cross_features也加入cat_features
    for (f1,f2) in CROSS_FEATURE_PAIRS:
        cross_name = f"{f1}_x_{f2}"
        if cross_name in all_cols:
            if cross_name not in cat_features:
                cat_features.append(cross_name)

    # time_of_day 放 cat_features
    if "time_of_day" in all_cols and "time_of_day" not in cat_features:
        cat_features.append("time_of_day")

    print(f"\n Total features: {len(cat_features) + len(num_features)}"
          f" {len(cat_features)} categorical, {len(num_features)} numeric"
          )
            
    return df, cat_features, num_features

def prepare_test_features(
        test_df: pd.DataFrame,
        cat_features: list[str],
        num_features: list[str],
) -> pd.DataFrame:
    """
    准备测试数据特征
    """
    # 同样变换处理
    test_df = engineer_time_features(test_df)
    test_df = decompose_location_and_category(test_df)

    # 保证所有特征都在
    expected_features = cat_features + num_features
    for col in expected_features:
        if col not in test_df.columns:
            print(f'column {col} not found in test_df.filling with 0')
            test_df[col] = 0

    # 筛选特征
    available = [c for c in expected_features if c in test_df.columns]
    X_test = test_df[available].copy()

    # 保证categorical columns 都是 string
    for col in cat_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
        
    return X_test
