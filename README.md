# Rainfall Prediction Classifier

A machine learning project that builds, optimizes, and compares multiple classification models to predict whether it will rain today in the Melbourne area of Australia. Uses a scikit-learn pipeline with GridSearchCV for hyperparameter tuning and includes rich exploratory data analysis and model evaluation visualizations.

---

## Project Overview

This project uses daily weather observations from the Australian Bureau of Meteorology (2008–2017) to predict daily rainfall. It compares three classifiers — **Random Forest**, **Logistic Regression**, and **Decision Tree** — evaluated across accuracy, precision, recall, F1-score, and AUC.

---

## Objectives

- Explore and perform feature engineering on a real-world weather dataset
- Build a classifier pipeline and optimize it using Grid Search Cross-Validation
- Evaluate models using performance metrics and visualizations
- Implement and compare multiple classifiers within the same pipeline
- Use an appropriate set of hyperparameters to search over for each model

---

## Dataset

- **Source:** [Australian Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/)
- **Kaggle:** [Weather Dataset - Rattle Package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/)
- **Coverage:** Daily weather observations across Australia from 2008 to 2017
- **Focus Area:** Melbourne, MelbourneAirport, and Watsonia (~7,500 records after filtering)

> Place `weatherAUS.csv` in the project root before running the notebook.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML pipelines, models, GridSearchCV |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scipy` | Bayesian posterior analysis |

---

## Feature Engineering

- Dropped rows with excessive missing values (Sunshine, Cloud cover)
- Renamed `RainToday` → `RainYesterday` and `RainTomorrow` → `RainToday` to reflect a same-day prediction framing
- Engineered a **Season** feature from the `Date` column (Summer / Autumn / Winter / Spring)
- Applied `StandardScaler` to numeric features and `OneHotEncoder` to categorical features inside a `ColumnTransformer`

---

## Models & Hyperparameter Grids

### Random Forest
```python
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}
```

### Logistic Regression
```python
param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}
```

### Decision Tree
```python
param_grid = {
    'classifier__max_depth': [5, 10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__class_weight': [None, 'balanced']
}
```

All models are optimized using **StratifiedKFold (5-fold)** cross-validation to preserve class distribution across folds.

---

## Visualizations

- Confusion Matrix (per model)
- Feature Importance Bar Chart (Random Forest)
- ROC Curve Comparison (all 3 models)
- Precision-Recall Curve (all 3 models)
- KDE Plot of Predicted Probabilities by True Label
- ACE Calibration Curves & Prediction Distribution
- Ridge Regression Diagnostic Plots
- Bayesian Posterior Distribution of Accuracy
- Learning Curves (all 3 models)
- Radar Chart — Multi-metric Model Comparison
- Rainfall Distribution by Season (count & percentage)
- Correlation Heatmap of Numeric Features
- Boxplots of Key Features vs Target

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ShahidAbas/rainfall-prediction-classifier.git
cd rainfall-prediction-classifier
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn seaborn scipy
```

### 3. Add the dataset
Download `weatherAUS.csv` from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/) and place it in the project root.

### 4. Run the notebook
```bash
jupyter notebook rainfallPredictionClassifier.ipynb
```
## License

This project is for educational purposes. Dataset credit: [Australian Bureau of Meteorology](http://www.bom.gov.au/) via Kaggle.
