# 广告转化率预测

## 项目简介

本项目基于广告点击日志数据，构建二分类模型预测用户是否会完成转化（如安装应用）。数据包含用户画像、广告素材、应用类别、广告位等多维度信息，训练集规模约 **370 万条**，正样本转化率约为 **2.58%**，属于典型的类别不平衡问题。

主要流程包括：
- 数据加载与清洗（内存优化至 ~187 MB）
- 探索性数据分析（EDA）
- 特征工程（时间特征、计数编码、目标编码、交叉特征）
- CatBoost + LightGBM 建模与对比
- Optuna 超参数优化
- 特征重要性与 SHAP 可解释性分析

---

## 数据集

数据文件位于 `./data/` 目录下：

| 文件名 | 描述 |
|--------|------|
| `train.csv` | 训练集（带 label） |
| `test.csv` | 测试集（无标签） |
| `ad.csv` | 广告信息（creativeID → adID, advertiserID, appID, appPlatform, campaignID） |
| `app_categories.csv` | 应用类别映射（appID → appCategory） |
| `position.csv` | 广告位信息（positionID → sitesetID, positionType） |
| `user.csv` | 用户画像（userID → 年龄、性别、学历、婚姻、生育、家乡、居住地） |
| `user_app_actions.csv` | 用户应用行为 |

### 主要特征

| 特征名 | 描述 |
|--------|------|
| `label` | 标签（0/1，是否转化） |
| `clickTime` | 点击时间（格式：DDHHMM） |
| `creativeID` | 广告素材 ID |
| `userID` | 用户 ID |
| `positionID` | 广告位 ID |
| `connectionType` | 联网方式 |
| `telecomsOperator` | 运营商 |
| `age`, `gender`, `education`, `marriageStatus`, `haveBaby` | 用户人口属性 |
| `hometown`, `residence` | 家乡、居住地（前两位省份，后两位城市） |
| `appID`, `appPlatform`, `appCategory` | 应用信息 |
| `sitesetID`, `positionType` | 广告位属性 |

---

## 项目结构

```
├── data/                         # 数据目录
├── src/                          # 源代码
│   ├── __init__.py
│   ├── config.py                 # 路径、常量、超参数配置
│   ├── data.py                   # 数据加载、清洗、合并、内存优化
│   ├── features.py               # 特征工程（编码、交叉特征等）
│   └── models.py                 # 模型训练、Optuna 调优、评估、SHAP
├── notebooks/                    # Jupyter Notebook
│   └── eda.ipynb                 # 探索性数据分析
├── output/                       # 输出文件（模型、图表、CSV）
├── main.py                       # 主入口（命令行运行）
├── requirements.txt              # Python 依赖
├── README.md                     # 项目说明（中文）
├── README.en.md                  # 项目说明（英文）
└── .gitignore
```

---

## 分析步骤

### 1. 数据清洗与加载

- 删除重复行（约 **57k** 条）
- 删除 `conversionTime` 列（97.5% 缺失）
- 合并 ad、user、app_categories、position 四张辅助表
- 基于 `user_app_actions` 提取用户行为特征（point-in-time，防数据泄露）
- 内存优化：676 MB → 187 MB（降幅 72.4%）

### 2. 探索性数据分析（EDA）

`notebooks/eda.ipynb` 包含：
- **类别不平衡诊断**：CVR 基线 ~2.58%
- **用户行为特征分布**：正负样本对比直方图、分位数 CVR 曲线
- **时间模式分析**：每日/每小时的点击量与 CVR 变化
- **多维度交叉 CVR 热力图**：connectionType × appPlatform 等
- **点二列相关系数**：快速筛选有用特征
- **高基数特征分析**：CVR 与样本量的 scatter plot

### 3. 特征工程

特征工程模块（`src/features.py`），最终产出 **58 维特征**（29 个类别 + 27 个数值）：

| 类别 | 说明 |
|------|------|
| **时间特征** | 从 `clickTime` 提取 `hours_since_start`、小时循环编码（sin/cos）、时段分桶（night/morning/afternoon/evening） |
| **地域拆分** | `hometown` → `hometown_province` + `hometown_city`，`residence` → `residence_province` + `residence_city` |
| **类目拆分** | `appCategory` → `first_category` + `second_category` |
| **计数编码** | 对 8 个高基数特征做 count + log-count encoding（仅用训练集统计，低频值 top 200 外归零防噪声） |
| **目标编码** | 时序交叉验证目标编码（TimeSeriesSplit，5-fold），防数据泄露 |
| **交叉特征** | 5 组低基数特征对交叉（如 `appPlatform × connectionType`） |

### 4. 建模与评估

按 `hours_since_start` 时间顺序切分 **70% 训练 / 30% 验证**（训练 2,584,473 条 / 验证 1,107,632 条）。

| 模型 | AUC | PR-AUC | F1 |
|------|-----|--------|-----|
| **CatBoost** | **0.7956** | **0.1128** | **0.1880** |
| LightGBM | 0.7904 | 0.1064 | 0.1822 |

CatBoost 在第 236 轮早停（subsample=0.85 增强了泛化，收敛更稳健），LightGBM 在第 269 轮早停。训练耗时约 45 分钟（含 SHAP）。

### 5. Optuna 超参数优化

通过 `--optuna` 启用，搜索空间：

| 参数 | 搜索范围 | 最终值 |
|------|----------|--------|
| `iterations` | 200–600 | 600 |
| `learning_rate` | 0.01–0.1（log 尺度） | 0.0365 |
| `depth` | 4–10 | 10 |
| `l2_leaf_reg` | 1–10（log 尺度） | 7.89 |
| `subsample` | 0.7–1.0 | 0.85（手工调低，增强泛化） |
| `colsample_bylevel` | 0.7–1.0 | 0.876 |

搜索策略：10 个 trial，用 30% 训练数据加速，MedianPruner 提前剪枝。最优参数已固化到 `config.py`，日常运行无需重复搜索。

### 6. SHAP 可解释性

通过 `--shap` 启用。对验证集采样 5000 条计算 SHAP 值。

**Top 10 SHAP 特征：**

| 特征 | SHAP 重要性 |
|------|-------------|
| `positionID_te` | 0.3750 |
| `appPlatform_x_connectionType` | 0.3545 |
| `userID` | 0.2782 |
| `sitesetID` | 0.0876 |
| `positionID` | 0.0696 |
| `creativeID_te` | 0.0678 |
| `age` | 0.0642 |
| `connectionType` | 0.0604 |
| `gender` | 0.0485 |
| `user_unique_apps_before` | 0.0451 |

> 目标编码（positionID_te）和交叉特征（appPlatform × connectionType）居首，说明模型核心信号来自业务逻辑组合和时序行为模式。raw userID 排名第三，与 count encoding 截断后的分布变化一致。

### 7. 输出文件

每次运行在 `output/` 下生成（⚠️ 仅 `catboost_model.cbm` 模型文件不纳入 Git，需本地运行生成；其余输出文件已纳入版本管理）：

| 文件 | 说明 |
|------|------|
| `catboost_model.cbm` | 训练好的 CatBoost 模型 |
| `catboost_curves.png` | CatBoost ROC + PR 曲线 |
| `catboost_feature_importance.csv` | CatBoost 特征重要性数据 |
| `catboost_feature_importance.png` | CatBoost 特征重要性柱状图 |
| `catboost_shap_summary.png` | SHAP 摘要图（需 `--shap`） |
| `catboost_shap_importance.csv` | SHAP 重要性数据（需 `--shap`） |
| `lightgbm_curves.png` | LightGBM ROC + PR 曲线 |
| `lightgbm_feature_importance.csv` | LightGBM 特征重要性数据 |
| `lightgbm_feature_importance.png` | LightGBM 特征重要性柱状图 |

---

## 如何运行

### 环境要求

- Python 3.10+
- Conda 环境（推荐）：`data_ana_project_env`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 命令行使用

```bash
# 基础流程：CatBoost + LightGBM
python main.py

# 启用 Optuna 超参数搜索
python main.py --optuna

# 启用 SHAP 可解释性
python main.py --shap

# 跳过 LightGBM（更快）
python main.py --skip-lightgbm

# 完整流程：Optuna + SHAP（CatBoost only）
python main.py --optuna --shap --skip-lightgbm

# 使用 conda 环境的 Python
D:\anaconda\envs\data_ana_project_env\python main.py --shap

# 若已将 conda 环境配置为默认 PATH，直接执行即可
python main.py --shap
```

### 参数说明

| 参数 | 作用 |
|------|------|
| `--optuna` | 启用 Optuna 超参数搜索 |
| `--shap` | 启用 SHAP 模型可解释性分析 |
| `--skip-lightgbm` | 跳过 LightGBM 对比训练 |
| `--predict` | 对 test.csv 生成预测（⚠️ 测试集特征缺失较多，不建议使用） |

---

## 改进方向

- 对 `userID` 等超高基数特征尝试负采样或 embedding
- 引入 Focal Loss 进一步处理类别不平衡
- LightGBM 的 Optuna 超参数搜索
- 尝试 XGBoost 作为第三对比模型
- 加载 `user_installedapps` 提取安装偏好的补充特征
- 构造更多交叉特征（如 `appCategory × positionType`）
