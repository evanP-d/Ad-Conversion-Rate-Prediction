# 广告转化率预测

## 项目简介

本项目基于广告点击日志数据，构建二分类模型预测用户是否会完成转化（如安装应用）。数据包含用户画像、广告素材、应用类别、广告位等多维度信息，训练集规模约 **370 万条**，正样本转化率约为 **2.49%**，属于典型的类别不平衡问题。

主要流程包括：
- 数据加载与清洗
- 探索性数据分析（EDA）
- 特征工程
- CatBoost 建模与评估
- 特征重要性分析

验证集 **AUC 达到 0.7720**。

---

## 数据集

数据文件位于 `./data/` 目录下：

| 文件名 | 描述 |
|--------|------|
| `train.csv` | 训练集（带标签 label） |
| `test.csv`  | 测试集（无标签） |
| `ad.csv`    | 广告信息（creativeID → adID, advertiserID, appID, appPlatform, campaignID） |
| `app_categories.csv` | 应用类别映射（appID → appCategory） |
| `position.csv` | 广告位信息（positionID → sitesetID, positionType） |
| `user.csv` | 用户画像（userID → 年龄、性别、学历、婚姻、生育、家乡、居住地） |
| `user_app_actions.csv` | 用户应用行为 |
| `user_installedapps.csv` | 用户已安装应用 |

### 主要特征

| 特征名 | 描述 |
|--------|------|
| `label` | 标签（0/1，是否转化） |
| `clickTime` | 点击时间（格式：DDHHMM） |
| `creativeID` | 广告素材ID |
| `userID` | 用户ID |
| `positionID` | 广告位ID |
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
├── 广告转化率预测.ipynb           # 完整分析笔记本
├── catboost_info/                # CatBoost 训练产物
├── README.md                     # 项目说明（中文）
├── README.en.md                  # 项目说明（英文）
└── .gitignore                    # Git 忽略规则
```

---

## 分析步骤

### 1. 数据清洗
- 删除重复行（约 **57k** 条）
- 删除 `conversionTime` 列（99% 为空）
- 检查其他数据表无缺失值/重复值

### 2. 探索性数据分析（EDA）

- **转化率基线**：2.49%（高度不平衡）
- **广告维度**：统计 `creativeID`、`adID`、`campaignID`、`advertiserID` 的转化率，识别高转化实体
- **用户维度**：分析年龄、性别、教育程度对转化率的影响
- **应用维度**：分析 `appID` 和 `appCategory` 的转化率
- **位置维度**：分析 `positionID`、`sitesetID` 的转化效果
- **时间维度**：分析每日/每小时的点击量与转化率分布

可视化使用 Matplotlib + Seaborn。

### 3. 特征工程

- **地域拆分**：`hometown` → `hometown_province`（省份）、`hometown_city`（城市）
- **应用类别拆分**：`appCategory` → `first_category`（一级类目）、`second_category`（二级类目）
- **时间特征**：从 `clickTime` 提取 `day`、`hour`、`minute`，构造连续变量 `hours_since_start`（从第0天0时起的小时数，含分钟小数）
- **删除冗余列**：`click_day`、`index` 等

最终特征数量：**25 个**（含数值特征与类别特征）

### 4. 建模与评估

**模型**：CatBoost（自动处理类别特征，支持类别不平衡）

**训练策略**：
- 按 `hours_since_start` 排序，按时间顺序划分：**70% 训练 / 30% 验证**
- 指定 **22 个类别特征**（包括 `creativeID`、`userID`、`positionID` 等）
- 超参数：
  - `iterations=500`
  - `learning_rate=0.03`
  - `depth=6`
  - `eval_metric='AUC'`
  - `early_stopping_rounds=50`

**结果**：
- 验证集 **AUC = 0.7720**

### 5. 特征重要性（Top 10）

| 特征 | 重要性 |
|------|--------|
| `connectionType` | 17.09 |
| `userID` | 12.16 |
| `positionID` | 10.18 |
| `sitesetID` | 10.13 |
| `appPlatform` | 9.23 |
| `campaignID` | 4.79 |
| `appID` | 4.68 |
| `first_category` | 4.23 |
| `residence` | 4.16 |
| `adID` | 3.80 |

> 说明：网络环境、用户身份、广告位对转化影响最大。

---

## 如何运行

### 环境要求
- Python 3.10+
- 依赖库：
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn catboost
  ```

### 运行步骤

1. 将所有数据文件放入 `./data/` 目录

2. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook 广告转化率预测.ipynb
   ```

3. 顺序执行所有单元格

### 结论

CatBoost 模型在时间顺序验证集上达到 **AUC 0.7720**，有效处理了高基数类别特征和类别不平衡问题。

最重要的特征是 `connectionType`（联网方式）、`userID`（用户标识）、`positionID` 与 `sitesetID`（广告位信息），说明**网络环境、用户行为和广告展示场景**是转化的关键驱动因素。

时间特征（`click_hour`、`hours_since_start`）贡献相对较小，但仍提供一定信息。地域和应用类目拆分有助于模型捕捉更细粒度的模式。

### 改进方向

- 尝试 LightGBM / XGBoost 对比效果
- 对高基数类别特征（如 `userID`）进行目标编码或负采样
- 构造交叉特征（例如 `appID × positionType`、`userID × adID`）
- 使用 SMOTE 或 Focal Loss 处理类别不平衡
- 超参数调优（网格搜索 / 贝叶斯优化）
