"""
数据加载 、清洗 、合并 + 提取用户行为特征
"""

import pandas as pd
import numpy as np



from .config import (
    TRAIN_PATH,
    TEST_PATH,
    AD_PATH,
    APP_CATEGORIES_PATH,
    POSITION_PATH,
    USER_PATH,
    USER_APP_ACTIONS_PATH,

    DROP_COLS,
    DTYPE_MAP,
    RANDOM_SEED,

)

def load_raw_data(use_test: bool = False) -> dict[str,pd.DataFrame]:
    """
    导入原数据
    """

    dfs = {}

    dfs['train'] = pd.read_csv(TRAIN_PATH, engine = 'python')
    
    if use_test:
        dfs['test'] = pd.read_csv(TEST_PATH, engine = 'python')
    
    dfs['ad'] = pd.read_csv(AD_PATH, engine = 'python')
    dfs['app_cat'] = pd.read_csv(APP_CATEGORIES_PATH, engine = 'python')
    dfs['position'] = pd.read_csv(POSITION_PATH, engine = 'python')
    dfs['user'] = pd.read_csv(USER_PATH, engine = 'python')

    # user_app_actions(6M ，3)
    uaa_chunks = []
    chunk_n = 0
    for chunk in pd.read_csv(USER_APP_ACTIONS_PATH,
                             engine = "python",
                             chunksize = 1000000):
        chunk_n += 1
        uaa_chunks.append(chunk)
    dfs['user_app_actions'] = pd.concat(
        uaa_chunks, ignore_index = True
    )
    del uaa_chunks

    return dfs

def clean_labeled_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    清洗训练数据
    """

    # 删重复行
    initial_rows = len(df)
    n_dupes = df.duplicated().sum()
    print(f"  Duplicated rows: {n_dupes}")

    df = df.drop_duplicates(keep="first")
    print(f"row after dedup: {len(df)}   ({initial_rows - len(df)} removed)")

    # 删无用列
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns = [col])
            print(f"  Drop {col} column")
    
    return df
    

def merge_auxiliary_tables(df: pd.DataFrame, dfs:dict[str,pd.DataFrame])->pd.DataFrame:
    """ 
    合并其余的额外表
    """
    if "adID" not in df.columns:
        df = pd.merge(df, dfs["ad"], on="creativeID", how="left")
        # 改一下camgaignID为campaignID
        if "camgaignID" in df.columns:
            df["campaignID"] = df["camgaignID"]
            df = df.drop(columns=["camgaignID"])
    

    if "age" not in df.columns: 
        df = pd.merge(df, dfs["user"], on="userID", how="left")

    
    if "appCategory" not in df.columns:
        df = pd.merge(df, dfs["app_cat"], on="appID", how="left")
    
    if "sitesetID" not in df.columns:
        df = pd.merge(df, dfs["position"], on="positionID", how="left")

    print(df.shape)
    return df


def extract_user_behavior_features(
        df:pd.DataFrame, dfs:dict[str,pd.DataFrame]
)->pd.DataFrame:
    """
    提取用户行为特征
    """
    # 数据本来就按照clickTime降序了，拆出click_day
    click_day = df['clickTime'].astype(str).str.zfill(6).str[:2].astype(int)

    # 把 user_app_actions的installTime拆出install_day
    uaa = dfs['user_app_actions'].copy()
    uaa['installTime'] = pd.to_numeric(uaa['installTime'],errors = 'coerce')
    uaa['install_day'] = uaa['installTime'].astype(str).str.zfill(6).str[:2].astype(int)

    # 通过聚合(userID,installday) 来分组计算安装的app数量 和 唯一的app数量
    daily = (
        uaa.groupby(["userID","install_day"]).agg(
            count = ("appID","count"),
            unique = ("appID","nunique"),
        )
    ).reset_index()

    # 通过排序来累加
    daily = daily.sort_values(by = ["userID","install_day"])
    daily['cum_count'] = daily.groupby("userID")["count"].cumsum()
    daily["cum_unique"] = daily.groupby("userID")['unique'].cumsum()

    # 偏移安装日，+1
    daily['ref_day'] = daily['install_day'] + 1

    # 保留需要的列
    daily = daily[['userID','ref_day','cum_count','cum_unique']]
    
    # 重命名列
    daily = daily.rename(
        columns = {
            "cum_count":"user_action_count_before",
            "cum_unique":"user_unique_apps_before",
        }
    )
    daily = daily.sort_values(by = ["userID","ref_day"], ignore_index = True)

    # 准备带有行编号的点击数据，用于时点关联
    df['click_day_parsed'] = click_day.values
    df['_row_id'] = range(len(df))

    # 时点关联合并，对每条点击记录都合并
    merged = df[['_row_id',"userID","click_day_parsed"]].merge(
        daily,
        on = "userID",
        how = "left",
    )

    # 过滤 ref_day <= click_day 把安装发生在点击后的筛掉
    merged = merged[merged['ref_day'] <= merged['click_day_parsed']]

    # row_id 和 ref_day也是笛卡尔积，选择保留筛选后ref_day最大的那个就行
    merged = merged.sort_values(['userID','ref_day'])

    pt_features = (
        merged.groupby('_row_id',sort=False).last().reset_index()
    )
    pt_features = pt_features[[
        "_row_id",
        'user_action_count_before',
        "user_unique_apps_before"
    ]]
    
    # 把结果按_row_id合回df
    df = df.merge(pt_features, on="_row_id", how="left")

    df['user_action_count_before'] = (
        df["user_action_count_before"].fillna(0).astype(np.float64)
    )

    df['user_unique_apps_before'] = (
        df["user_unique_apps_before"].fillna(0).astype(np.float64)
    )

    df = df.drop(columns = ['_row_id'])
    del merged, pt_features, daily, uaa

    # 把click_day_parsed也删掉
    df = df.drop(columns = ['click_day_parsed'])
    return df

def optimize_memory(df:pd.DataFrame)->pd.DataFrame:
    """
    优化内存占用
    """

    before_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Before optimization: {before_mb:.2f} MB")

    for col in df.columns:
        if col in DTYPE_MAP:
            try:
                df[col] = df[col].astype(DTYPE_MAP[col])
            except (ValueError, TypeError):
                pass
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col],downcast = 'integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast = 'float')
    
    after_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  After optimization: {after_mb:.2f} MB")

    return df



def load_and_prepare_data(use_test:bool = False) -> dict:
    """
    流水线 数据加载、清洗、合并、特征提取
    """
    np.random.seed(RANDOM_SEED)

    # load
    dfs = load_raw_data(use_test = use_test)

    # clean
    df = clean_labeled_data(dfs['train'])

    # merge
    df = merge_auxiliary_tables(df, dfs)

    # user behavior features
    df = extract_user_behavior_features(df, dfs)

    # memory optimization
    df = optimize_memory(df)

    # 处理test 数据
    test_df = None
    if use_test and "test" in dfs:
        test_df = dfs['test'].copy()
        test_df = merge_auxiliary_tables(test_df, dfs)
        test_df = extract_user_behavior_features(test_df, dfs)
        test_df = optimize_memory(test_df)
    
    if test_df is not None:
        print(f"test shape: {test_df.shape}")

    return {'train_val':df, "test":test_df}


