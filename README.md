# Exoplanet Habitability Prediction

An end-to-end machine-learning project for identifying exoplanet candidates that lie within the habitable zone of their host star.

The project compares Logistic Regression, Support Vector Machine, and XGBoost pipelines under class imbalance. It includes feature engineering, threshold tuning, SMOTE experiments, model interpretation, cross-model comparison, and validation checks.

## Project question

Can available exoplanet and host-star measurements be used to classify whether an observation belongs to the habitable-zone class?

- **Target:** `habitable_zone_flag` (binary classification)
- **Observation unit:** one exoplanet observation
- **Primary evaluation focus:** PR-AUC and F1, because the positive class is highly imbalanced
- **Secondary metrics:** ROC-AUC, precision, recall, calibration, and train/test performance gaps

This is a predictive classification project. Feature importance indicates model association and predictive usefulness; it does not establish that a feature is a causal driver of habitability.

## Dataset

The data are based on the [NASA Exoplanet Archive Intelligence dataset on Kaggle](https://www.kaggle.com/datasets/kanchana1990/nasa-exoplanet-archive-intelligence).

The repository currently includes:

- `data/nasa_exoplanet_intelligence.csv`: source dataset used for the cleaning workflow
- `data/exoplanet_clean.csv`: cleaned dataset used by the three independent modelling notebooks

## Repository structure

```text
.
├── data/
│   ├── nasa_exoplanet_intelligence.csv
│   └── exoplanet_clean.csv
├── 0050ML_Logistic_Regression.ipynb
├── 0050ML_SVM.ipynb
├── 0050ML_XGB.ipynb
├── COMP0050_complete.ipynb
├── requirements.txt
└── README.md
```

The three model notebooks are intentionally kept as independent analyses. `COMP0050_complete.ipynb` contains the combined workflow from cleaning and EDA through model comparison and validation.

## Methodology

### Data preparation and EDA

- Missing-value handling and outlier inspection
- Class-imbalance analysis
- Distribution, correlation, mutual-information, and PCA analysis
- Log transformations and domain-informed derived features
- Numeric, categorical, and boolean feature handling

### Model experiments

| Model family | Main experiments |
|---|---|
| Logistic Regression | Baseline, regularisation tuning, threshold tuning, SMOTE |
| SVM | Baseline, SMOTE, grid search, threshold tuning |
| XGBoost | Baseline, hyperparameter tuning, threshold tuning, SHAP interpretation |

The combined workflow uses stratified 60/20/20 train/validation/test splits. Hyperparameter and threshold selection are performed before final evaluation on the untouched test set. SMOTE is applied only to training data or inside cross-validation pipelines.

## Current benchmark

The following values are recorded in the current executed output of `COMP0050_complete.ipynb`:

| Model | PR-AUC | F1 | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression, tuned + threshold | 0.427 | 0.577 | 0.947 | 0.456 | **0.785** |
| SVM, grid + SMOTE + threshold | **0.788** | **0.671** | **0.977** | **0.647** | 0.696 |
| XGBoost, tuned + threshold | 0.644 | 0.663 | 0.971 | 0.632 | 0.696 |

On the saved test-set output, SVM provides the strongest PR-AUC and F1 balance, while Logistic Regression provides the highest recall at the selected threshold. These figures should be regenerated after changing dependencies or rerunning the notebooks.

## Interpretation and limitations

- Class imbalance makes accuracy alone unsuitable as the primary metric.
- Threshold tuning changes the precision/recall trade-off and must be selected on validation data only.
- `stellar_flux` and related engineered variables are simplified proxies rather than complete physical models.
- The dataset and labels are dependent on the source data definition and preprocessing assumptions.
- SHAP and permutation importance explain model behaviour; they should not be interpreted as causal evidence.
- The project is an offline modelling study and does not include deployment monitoring or prospective validation.

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Recommended order:

1. Run `COMP0050_complete.ipynb` to reproduce cleaning, EDA, feature engineering, model comparison, and validation.
2. Run `0050ML_Logistic_Regression.ipynb`, `0050ML_SVM.ipynb`, or `0050ML_XGB.ipynb` independently when you want to inspect one model family.

The independent notebooks read `data/exoplanet_clean.csv`. The complete notebook reads the raw file from `data/` and regenerates the cleaned file at the same location.

## Skills

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, imbalanced-learn, SMOTE, feature engineering, imbalanced classification, threshold optimisation, model evaluation, data visualisation, and reproducible notebook workflows.

## Authors

Stephan Vermeulen, Guanwen Zhou, Callum Harris, Ziqi Deng, and Zhicong Deng.
