# Ad Conversion Rate Prediction

## Project Overview

This project builds a binary classification model to predict whether a user will complete a conversion (e.g., app installation) based on ad click log data. The dataset contains multi-dimensional information including user profiles, ad creatives, app categories, and ad placements. The training set contains approximately **3.7 million** records with a positive conversion rate of approximately **2.58%**, representing a typical class imbalance problem.

The main workflow includes:
- Data loading and cleaning (memory-optimized to ~187 MB)
- Exploratory Data Analysis (EDA)
- Feature engineering (time features, count encoding, target encoding, cross features)
- CatBoost + LightGBM modeling and comparison
- Optuna hyperparameter optimization
- Feature importance and SHAP interpretability analysis

---

## Dataset

Data files are located in the `./data/` directory:

| File | Description |
|------|-------------|
| `train.csv` | Training set (with label) |
| `test.csv` | Test set (without label) |
| `ad.csv` | Ad information (creativeID → adID, advertiserID, appID, appPlatform, campaignID) |
| `app_categories.csv` | App category mapping (appID → appCategory) |
| `position.csv` | Ad placement info (positionID → sitesetID, positionType) |
| `user.csv` | User profiles (userID → age, gender, education, marital status, parental status, hometown, residence) |
| `user_app_actions.csv` | User app behavior data |

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
| `hometown`, `residence` | Hometown and residence (province + city encoded) |
| `appID`, `appPlatform`, `appCategory` | App information |
| `sitesetID`, `positionType` | Ad placement attributes |

---

## Project Structure

```
├── data/                         # Raw data files
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Paths, constants, hyperparameters
│   ├── data.py                   # Data loading, cleaning, merging, memory optimization
│   ├── features.py               # Feature engineering (encodings, cross features)
│   └── models.py                 # Model training, Optuna tuning, evaluation, SHAP
├── notebooks/                    # Jupyter Notebook
│   └── eda.ipynb                 # Exploratory Data Analysis
├── output/                       # Output (models, plots, CSVs)
├── main.py                       # Main entry point (CLI)
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation (Chinese)
├── README.en.md                  # Documentation (English)
└── .gitignore
```

---

## Analysis Steps

### 1. Data Cleaning & Loading

- Remove duplicate rows (~**57k** records)
- Drop `conversionTime` column (97.5% missing)
- Merge four auxiliary tables: ad, user, app_categories, position
- Extract user behavior features from `user_app_actions` (point-in-time, no data leakage)
- Memory optimization: 676 MB → 187 MB (72.4% reduction)

### 2. Exploratory Data Analysis (EDA)

`notebooks/eda.ipynb` includes:
- **Class imbalance diagnosis**: CVR baseline ~2.58%
- **User behavior feature distribution**: positive vs. negative histograms, decile CVR curves
- **Time pattern analysis**: daily/hourly click volume and CVR trends
- **Cross-dimensional CVR heatmaps**: connectionType × appPlatform, etc.
- **Point-biserial correlation**: quick feature screening
- **High-cardinality feature analysis**: CVR vs. volume scatter plots

### 3. Feature Engineering

Feature engineering module (`src/features.py`), producing **58 features** (29 categorical + 27 numeric):

| Category | Description |
|----------|-------------|
| **Time features** | Extract `hours_since_start` from `clickTime`, cyclic hour encoding (sin/cos), time-of-day buckets (night/morning/afternoon/evening) |
| **Region decomposition** | `hometown` → `hometown_province` + `hometown_city`, `residence` → `residence_province` + `residence_city` |
| **Category decomposition** | `appCategory` → `first_category` + `second_category` |
| **Count encoding** | Count + log-count encoding on 8 high-cardinality features (fit on training set only, top-200 values kept, tail values zeroed to reduce noise) |
| **Target encoding** | Time-series cross-validation target encoding (TimeSeriesSplit, 5-fold), leakage-free |
| **Cross features** | 5 low-cardinality feature pair crosses (e.g., `appPlatform × connectionType`) |

### 4. Modeling and Evaluation

Chronological split by `hours_since_start`: **70% training / 30% validation** (2,584,473 training / 1,107,632 validation records).

| Model | AUC | PR-AUC | F1 |
|------|-----|--------|-----|
| **CatBoost** | **0.7956** | **0.1128** | **0.1880** |
| LightGBM | 0.7904 | 0.1064 | 0.1822 |

CatBoost early-stopped at iteration 236 (subsample=0.85 improves generalization, slower but more robust convergence), LightGBM at iteration 269. Training time ~45 minutes (including SHAP).

### 5. Optuna Hyperparameter Optimization

Enabled via `--optuna` flag. Search space and final values:

| Parameter | Search Range | Final Value |
|-----------|-------------|-------------|
| `iterations` | 200–600 | 600 |
| `learning_rate` | 0.01–0.1 (log scale) | 0.0365 |
| `depth` | 4–10 | 10 |
| `l2_leaf_reg` | 1–10 (log scale) | 7.89 |
| `subsample` | 0.7–1.0 | 0.85（manually lowered for better generalization） |
| `colsample_bylevel` | 0.7–1.0 | 0.876 |

Strategy: 10 trials, 30% training data sampling for speed, MedianPruner for early pruning. Best parameters are saved in `config.py` so daily runs skip the search.

### 6. SHAP Interpretability

Enabled via `--shap` flag. Computes SHAP values on 5,000 validation samples.

**Top 10 SHAP Features:**

| Feature | SHAP Importance |
|---------|----------------|
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

> Target encoding (positionID_te) and cross features (appPlatform × connectionType) dominate — the model's core signals come from business logic combinations and temporal behavior patterns. Raw userID ranks third, consistent with the count encoding tail truncation shifting feature distribution.

### 7. Output Files

Generated in `output/` on each run (⚠️ only the `catboost_model.cbm` model file is not tracked in Git; run the pipeline locally to generate it; other output files are tracked in version control):

| File | Description |
|------|-------------|
| `catboost_model.cbm` | Trained CatBoost model |
| `catboost_curves.png` | CatBoost ROC + PR curves |
| `catboost_feature_importance.csv` | CatBoost feature importance data |
| `catboost_feature_importance.png` | CatBoost feature importance bar chart |
| `catboost_shap_summary.png` | SHAP summary plot (`--shap` required) |
| `catboost_shap_importance.csv` | SHAP importance data (`--shap` required) |
| `lightgbm_curves.png` | LightGBM ROC + PR curves |
| `lightgbm_feature_importance.csv` | LightGBM feature importance data |
| `lightgbm_feature_importance.png` | LightGBM feature importance bar chart |

---

## How to Run

### Requirements

- Python 3.10+
- Conda environment (recommended): `data_ana_project_env`

### Install Dependencies

```bash
pip install -r requirements.txt
```

### CLI Usage

```bash
# Basic pipeline: CatBoost + LightGBM
python main.py

# Enable Optuna hyperparameter search
python main.py --optuna

# Enable SHAP interpretability
python main.py --shap

# Skip LightGBM (faster)
python main.py --skip-lightgbm

# Full pipeline: Optuna + SHAP (CatBoost only)
python main.py --optuna --shap --skip-lightgbm

# Using conda environment Python
/d/anaconda/envs/data_ana_project_env/python main.py --shap
```

### CLI Flags

| Flag | Effect |
|------|--------|
| `--optuna` | Enable Optuna hyperparameter optimization |
| `--shap` | Enable SHAP model interpretability analysis |
| `--skip-lightgbm` | Skip LightGBM comparison training |
| `--predict` | Generate test set predictions (⚠️ not recommended — many features missing in test set) |

---

## Future Improvements

- Negative sampling or embedding for ultra-high-cardinality features like `userID`
- Focal Loss for better class imbalance handling
- Optuna hyperparameter search for LightGBM
- Add XGBoost as a third comparison model
- Load `user_installedapps` to extract additional install-preference features
- More cross features (e.g., `appCategory × positionType`)
