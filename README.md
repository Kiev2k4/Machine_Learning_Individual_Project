# Life Expectancy Prediction Model 🌍

**Machine Learning Academic Project** — A regression pipeline that predicts national life expectancy using the WHO Life Expectancy dataset, covering 193 countries across 16 years. Built with Python, scikit-learn, and pandas as part of COSC2753 Machine Learning.

> 📚 **Academic Project** — 2025 | RMIT University

---

## Overview

This project frames life expectancy as a supervised regression problem. Starting from a messy real-world dataset with extensive missing values across 20+ health and socioeconomic indicators, the notebook walks through a complete ML workflow: exploratory analysis, feature-informed imputation, model training, hyperparameter tuning, and final evaluation across five regression architectures.

**Target variable:** `TARGET_LifeExpectancy` — national life expectancy in years (continuous)

---

## Features

### 📊 Exploratory Data Analysis
- Distribution plots and boxplots for all numerical features
- Skewness analysis to identify heavy-tailed variables
- Full correlation heatmap across all 20+ features
- Scatterplots of the strongest predictors against life expectancy
- Key findings: `IncomeCompositionOfResources`, `Schooling`, and `AdultMortality` emerge as the highest-correlation features

### 🔧 Feature Engineering & Imputation

Rather than using simple mean/median fill for every column, the notebook uses **regression-based imputation** wherever correlated features are available — preserving inter-variable relationships rather than distorting distributions.

| Feature | Imputation Strategy |
|---|---|
| `AdultMortality` | Mean of `AdultMortality-Male` and `AdultMortality-Female` |
| `Alcohol` | Median fill |
| `GDP` ↔ `PercentageExpenditure` | Mutual linear regression (each predicts the other) |
| `Under5LS` | Filled from `SLS` where missing |
| `Measles` | Linear regression from `Under5LS` + `SLS` |
| `Polio` | Linear regression from `Diphtheria` |
| `TotalExpenditure` | Linear regression from `Status`, `Alcohol`, `BMI` |
| `Thinness1-19years` ↔ `Thinness5-9years` | Mutual linear regression |
| `IncomeCompositionOfResources` ↔ `Schooling` | Mutual linear regression |
| `HIV-AIDS` | Linear regression from adult mortality features |
| `Population` | Linear regression from `SLS` + `Under5LS` |
| `BMI` | Linear regression from life expectancy, income composition, schooling |

### 🤖 Model Training & Evaluation

Five regression architectures are trained, validated, and tested:

| Model | Notes |
|---|---|
| **Linear Regression** | Baseline model |
| **Ridge Regression** | GridSearchCV over α ∈ {0.1, 1.0, 10.0, 100.0}; best α = 0.1 |
| **Lasso Regression** | RandomizedSearchCV, α sampled from Uniform(0.01, 10); best α ≈ 0.216 |
| **Polynomial Regression** | Degree-2 features via `PolynomialFeatures`, sklearn Pipeline |
| **Random Forest** | 100 estimators, random_state = 42 |

All models are evaluated with three metrics: **MAE**, **RMSE**, and **R²**

### 📐 Data Splitting & Scaling
- **60 / 20 / 20** train / validation / test split (stratified by random seed 42)
- **StandardScaler** fitted exclusively on the training set, applied to validation and test

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data loading, manipulation, imputation |
| NumPy | Numerical operations |
| scikit-learn | Preprocessing, models, hyperparameter tuning, metrics |
| Matplotlib / Seaborn | EDA visualisations and correlation heatmap |
| joblib | Model serialisation (`*.pkl`) |
| Jupyter Notebook | Interactive development environment |

---

## Dataset

**WHO Life Expectancy dataset** — multi-country, multi-year longitudinal health data.

| Split | File | Rows |
|---|---|---|
| Train | `train.csv` | ~2,071 |
| Test | `test.csv` | ~867 |

**22 input features** covering: adult mortality (overall, male, female), alcohol consumption, health expenditure, immunisation rates (Polio, Diphtheria, Measles), BMI, under-5 and infant survival, HIV/AIDS prevalence, GDP, population, thinness prevalence, income composition of resources, and schooling years.

`Country`, `Year`, `Status`, and `ID` are dropped before modelling — the model learns from health and economic indicators rather than country identity.

---

## Project Structure

```
COSC2753_A1_s3978175/
├── train.ipynb                          # Full notebook: EDA → imputation → models → evaluation
├── train.csv                            # WHO training data
├── test.csv                             # WHO test data (no target column)
├── COSC2753_A1_Predictions_s3978175.csv # Final predictions on test set
├── linear_model.pkl                     # Serialised Linear Regression model
├── ridge_model.pkl                      # Serialised Ridge Regression model
└── scaler.pkl                           # Fitted StandardScaler
```

---

## Getting Started

### Prerequisites
- Python >= 3.8
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib scipy
```

### Run the notebook

1. Place `train.csv` and `test.csv` in the same directory as `train.ipynb`
2. Launch Jupyter:

```bash
jupyter notebook train.ipynb
```

3. Run all cells — the notebook executes end-to-end from raw CSV to final predictions

---

## Methodology

```
Raw CSV
  │
  ├── EDA (distributions, correlations, outliers)
  │
  ├── Regression-based imputation (12 features)
  │
  ├── Feature selection (drop ID, Country, Year, Status)
  │
  ├── 60/20/20 train/val/test split
  │
  ├── StandardScaler (fit on train only)
  │
  ├── Model training
  │     ├── Linear Regression
  │     ├── Ridge  (GridSearchCV, 5-fold CV)
  │     ├── Lasso  (RandomizedSearchCV, 5-fold CV)
  │     ├── Polynomial (degree=2 pipeline)
  │     └── Random Forest (100 estimators)
  │
  ├── Validation-set comparison (MAE, RMSE, R²)
  │
  └── Test-set final evaluation → predictions CSV
```

---

## Key Findings from EDA

`IncomeCompositionOfResources` and `Schooling` show the strongest positive correlations with life expectancy, reflecting the well-established link between human development and longevity. `AdultMortality` and `HIV-AIDS` prevalence are the strongest negative predictors. Immunisation rates (Polio, Diphtheria) correlate positively but less strongly — likely because they proxy general healthcare infrastructure rather than acting as direct causes.

---

## Author

**Hau Nguyen**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hau-nguyen-521233254/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Kiev2k4)
