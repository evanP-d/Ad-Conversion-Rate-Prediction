# Ad Conversion Rate Prediction

## Project Overview

This project builds a binary classification model to predict whether a user will complete a conversion (e.g., app installation) based on ad click log data. The dataset contains multi-dimensional information including user profiles, ad creatives, app categories, and ad placements. The training set contains approximately **3.7 million** records with a positive conversion rate of approximately **2.49%**, representing a typical class imbalance problem.

The main workflow includes:
- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- CatBoost modeling and evaluation
- Feature importance analysis

The validation set achieves an **AUC of 0.7720**.

---

## Dataset

Data files are located in the `./data/` directory:

| File | Description |
|------|-------------|
| `train.csv` | Training set (with label) |
| `test.csv`  | Test set (without label) |
| `ad.csv`    | Ad information (creativeID → adID, advertiserID, appID, appPlatform, campaignID) |
| `app_categories.csv` | App category mapping (appID → appCategory) |
| `position.csv` | Ad placement info (positionID → sitesetID, positionType) |
| `user.csv` | User profiles (userID → age, gender, education, marital status, parental status, hometown, residence) |
| `user_app_actions.csv` | User app behavior data |
| `user_installedapps.csv` | User installed apps data |

### Key Features

| Feature | Description |
|---------|-------------|
| `label` | Target (0/1, whether conversion occurred) |
| `clickTime` | Click timestamp (format: DDHHMM) |
| `creativeID` | Ad creative ID |
| `userID` | User ID |
| `positionID` | Ad placement ID |
| `connectionType` | Network connection type |
| `telecomsOperator` | Telecom operator |
| `age`, `gender`, `education`, `marriageStatus`, `haveBaby` | User demographic attributes |
| `hometown`, `residence` | Hometown and residence (first 2 digits: province, last 2 digits: city) |
| `appID`, `appPlatform`, `appCategory` | App information |
| `sitesetID`, `positionType` | Ad placement attributes |

---

## Project Structure

```
├── data/                                  # Data directory
├── 广告转化率预测.ipynb                    # Complete analysis notebook
├── catboost_info/                         # CatBoost training artifacts
├── README.md                              # Project documentation (Chinese)
├── README.en.md                           # Project documentation (English)
└── .gitignore                             # Git ignore rules
```

---

## Analysis Steps

### 1. Data Cleaning
- Remove duplicate rows (~**57k** records)
- Drop `conversionTime` column (99% missing)
- Verify other data tables have no missing or duplicate values

### 2. Exploratory Data Analysis (EDA)

- **Conversion rate baseline**: 2.49% (highly imbalanced)
- **Ad dimension**: Analyze conversion rates by `creativeID`, `adID`, `campaignID`, `advertiserID` to identify high-converting entities
- **User dimension**: Analyze the impact of age, gender, and education on conversion rates
- **App dimension**: Analyze conversion rates by `appID` and `appCategory`
- **Placement dimension**: Analyze conversion performance by `positionID` and `sitesetID`
- **Time dimension**: Analyze daily/hourly click volume and conversion rate distribution

Visualizations created with Matplotlib + Seaborn.

### 3. Feature Engineering

- **Region splitting**: `hometown` → `hometown_province` (province), `hometown_city` (city)
- **App category splitting**: `appCategory` → `first_category` (level 1), `second_category` (level 2)
- **Time features**: Extract `day`, `hour`, `minute` from `clickTime`, construct continuous variable `hours_since_start` (hours from day 0, hour 0, including fractional minutes)
- **Remove redundant columns**: `click_day`, `index`, etc.

Final feature count: **25** (including both numerical and categorical features)

### 4. Modeling and Evaluation

**Model**: CatBoost (automatically handles categorical features and supports class imbalance)

**Training Strategy**:
- Sort by `hours_since_start`, split chronologically: **70% training / 30% validation**
- Specify **22 categorical features** (including `creativeID`, `userID`, `positionID`, etc.)
- Hyperparameters:
  - `iterations=500`
  - `learning_rate=0.03`
  - `depth=6`
  - `eval_metric='AUC'`
  - `early_stopping_rounds=50`

**Result**:
- Validation set **AUC = 0.7720**

### 5. Feature Importance (Top 10)

| Feature | Importance |
|---------|------------|
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

> Note: Network environment, user identity, and ad placement have the greatest impact on conversion.

---

## How to Run

### Requirements
- Python 3.10+
- Dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn catboost
  ```

### Steps
1. Place all data files in the `./data/` directory

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook 广告转化率预测.ipynb
   ```

3. Execute all cells in order

### Conclusion

The CatBoost model achieved an AUC of 0.772 on a chronologically-split validation set, effectively handling high-cardinality categorical features and class imbalance.

The most important features are `connectionType` (network type), `userID` (user identifier), `positionID` and `sitesetID` (ad placement info), indicating that **network environment, user behavior, and ad display context** are the key drivers of conversion.

Time-based features (`click_hour`, `hours_since_start`) contribute relatively less but still provide useful information.

Region and app category splitting helps the model capture more fine-grained patterns.

### Future Improvements
- Compare with LightGBM / XGBoost
- Apply target encoding or negative sampling for high-cardinality categorical features (e.g., `userID`)
- Construct cross features (e.g., `appID × positionType`, `userID × adID`)
- Use SMOTE or Focal Loss to handle class imbalance
- Hyperparameter tuning (Grid Search / Bayesian Optimization)
