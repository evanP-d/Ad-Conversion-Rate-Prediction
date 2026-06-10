""" 
配置模块 - 用于定义所有路径、常量 和 超参数
1. 路径配置：数据文件路径、输出文件路径
2. 数据处理配置：时间切分比例、随机种子
3. 特征定义：所有特征列表、高基数特征、交叉特征对
4. 模型配置：模型参数
"""
import os


# 项目根目录、数据目录、输出目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据文件路径
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

AD_PATH = os.path.join(DATA_DIR, "ad.csv")
APP_CATEGORIES_PATH = os.path.join(DATA_DIR, "app_categories.csv")

POSITION_PATH = os.path.join(DATA_DIR, "position.csv")

USER_PATH = os.path.join(DATA_DIR, "user.csv")
USER_APP_ACTIONS_PATH = os.path.join(DATA_DIR, "user_app_actions.csv")

# 输出文件路径
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "catboost_model.cbm")



# 时间切分比例 
SPLIT_RATIO = 0.7

# 随机种子
RANDOM_SEED = 42

# count encoding的最大类别数，超过则归为"other"
MAX_CAT_COUNT = 200




# 数据类型映射（用于内存优化）    之前没优化做交叉衍生，内存给我干爆了 
DTYPE_MAP = {
    "label" : "int8",
    "clickTime": "int32",
    "creativeID" : "int32",
    "userID": "int32",
    "positionID": "int32",
    "connectionType": "int8",
    "telecomsOperator": "int8",
    "adID": "int32",
    "advertiserID": "int16",
    "appID": "int16",
    "appPlatform": "int8",
    "campaignID": "int32",
    "age": "int8",
    "gender": "int8",
    "education": "int8",
    "marriageStatus": "int8",
    "haveBaby": "int8",
    "residence_province": "int16",
    "residence_city": "int16",
    "sitesetID": "int8",
    "positionType": "int8",
    "hometown_province": "int16",
    "hometown_city": "int16",
}


# 删除的列
DROP_COLS = ["conversionTime"]


# 数值特征（连续特征）
NUMERIC_FEATURES = [
    "age",
    "hours_since_start",
]

# 分类特征 建模使用的
CATEGORICAL_FEATURES = [
    "creativeID",
    "userID",
    "positionID",
    "connectionType",
    "telecomsOperator",
    "adID",
    "advertiserID",
    "appID",
    "appPlatform",
    "campaignID",
    "gender",
    "education",
    "marriageStatus",
    "haveBaby",
    "sitesetID",
    "positionType",
    "hometown_province",
    "hometown_city",
    "residence_province",
    "residence_city",
    "first_category",
    "second_category",
    ]


# 目标编码高基数特征
HIGH_CARDINALITY_FEATURES = [
    "userID",
    "creativeID",
    "positionID",
    "adID",
    "campaignID",
    "advertiserID",
    "appID",

]

# 交叉特征对 （低基数）
CROSS_FEATURE_PAIRS = [
    ("appPlatform","connectionType"),
    ("positionType","sitesetID"),
    ("advertiserID","first_category"),
    ("positionType","first_category"),
    ("connectionType","telecomsOperator")
]


# # Catboost超参数
# CATBOOST_PARAMS = {
#     "iterations":1000,
#     "learning_rate":0.03,
#     "depth":6,
#     "loss_function":"Logloss",
#     "eval_metric":"AUC",
#     "random_seed":RANDOM_SEED,
#     "verbose":100,
#     "early_stopping_rounds":50,
#     "auto_class_weights":"Balanced",
#     "thread_count":-1,  # 使用所有CPU核心
# }

# optuna优化后的catboost超参数
CATBOOST_PARAMS = {
    "iterations":600,
    "learning_rate":0.03649155981381801,
    "depth":10,
    "l2_leaf_reg":7.886332236196301,
    "subsample":0.85,
    "colsample_bylevel":0.875546232291729292,
    "loss_function":"Logloss",
    "eval_metric":"AUC",
    "random_seed":RANDOM_SEED,
    "verbose":100,
    "early_stopping_rounds":50,
    "auto_class_weights":"Balanced",
    "thread_count":-1,  # 使用所有CPU核心
}


# lightgbm超参数
LIGHTGBM_PARAMS = {
    "objective":"binary",
    "metric":"auc",
    "boosting_type":"gbdt",
    "learning_rate":0.03,
    "num_leaves":63,
    "max_depth":6,
    "feature_fraction":0.8, # 随机选择80%的特征进行每轮训练，防止过拟合
    "bagging_fraction":0.8, # 随机选择80%的数据进行每轮训练，防止过拟合
    "bagging_freq":5, # 每5轮进行一次bagging
    "min_data_in_leaf":50,
    "scale_pos_weight":20, # 处理类别不平衡，正负样本比例约为1:40
    "verbose":-1, # 关闭LightGBM的日志输出
    "random_state":RANDOM_SEED,
    "n_jobs":-1,
}


# ----------------------------optuna search config-----------------------

OPTUNA_TRIALS = 10
OPTUNA_TIMEOUT = 1800