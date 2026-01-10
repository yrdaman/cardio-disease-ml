# 🫀 Cardiovascular Disease Risk Prediction

A machine learning project for early cardiovascular disease risk screening, optimized for **high recall** to minimize missed diagnoses in healthcare settings.

---

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting cardiovascular disease risk. The system is designed as a **decision-support tool** for healthcare screening, prioritizing the identification of at-risk patients over overall accuracy.

**Key Features:**
- Recall-optimized Gradient Boosting model with probability calibration
- Custom decision threshold (0.36) tuned for high sensitivity
- Interactive Streamlit web application for real-time predictions
- Automated BMI calculation from height and weight inputs
- Professional preprocessing pipeline with StandardScaler

---

## Problem Statement

Cardiovascular disease (CVD) is one of the leading causes of death globally. Early detection and risk assessment are critical for preventive care. In medical screening scenarios, **missing a positive case (false negative) is more costly than a false alarm (false positive)**.

This project addresses the need for:
- High-sensitivity risk screening to catch potential CVD cases
- A user-friendly interface for healthcare decision support
- Transparent probability scores rather than binary predictions

---

## Dataset Description

**Source:** Kaggle Cardiovascular Disease Dataset  
**Records:** ~70,000 patient records  
**Target Variable:** `cardio` (1 = cardiovascular disease present, 0 = absent)

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Continuous |
| `gender` | 0 = Female, 1 = Male | Categorical |
| `height` | Height in cm | Continuous |
| `bmi` | Body Mass Index (calculated) | Continuous |
| `ap_hi` | Systolic blood pressure | Continuous |
| `ap_lo` | Diastolic blood pressure | Continuous |
| `cholesterol` | 1 = Normal, 2 = Above Normal, 3 = Well Above Normal | Ordinal |
| `gluc` | Glucose level (1-3 scale) | Ordinal |
| `smoke` | Smoking status (0/1) | Binary |
| `alco` | Alcohol intake (0/1) | Binary |
| `active` | Physical activity (0/1) | Binary |

**Preprocessing Applied:**
- Removed `id` column and converted age from days to years
- Replaced `weight` with calculated `BMI` feature
- Cleaned outliers in blood pressure readings
- Final processed dataset: ~65,000 records with 11 features

---

## Machine Learning Pipeline

### 1. Data Preprocessing
```
Raw Data → Outlier Removal → Feature Engineering (BMI) → Train/Test Split (80/20) → StandardScaler
```

- **Stratified split** to maintain class balance
- **StandardScaler** fitted only on training data to prevent data leakage
- Scaler is bundled with the trained model inside models/final_model.pkl

### 2. Baseline Model Comparison

Multiple algorithms evaluated with focus on **recall**:

| Model | Recall |
|-------|--------|
| Logistic Regression | ~0.70 |
| K-Nearest Neighbors | ~0.65 |
| Support Vector Machine | ~0.68 |
| Decision Tree | ~0.62 |
| Random Forest | ~0.68 |
| Gradient Boosting | ~0.72 |

### 3. Hyperparameter Tuning

**GridSearchCV** with 5-fold cross-validation optimized for recall:

**Logistic Regression:**
```python
param_grid = {'C': [0.01, 0.1, 1, 10]}
# class_weight='balanced'
```

**Gradient Boosting (Selected):**
```python
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}
```

---

## Model Training and Optimization

### Final Model: Calibrated Gradient Boosting Classifier

The Gradient Boosting model was selected based on:
- Highest baseline recall among ensemble methods
- Good feature importance interpretability
- Stable performance across cross-validation folds

### Probability Calibration

Applied **Isotonic Calibration** using `CalibratedClassifierCV` to ensure predicted probabilities are reliable and well-calibrated:

```python
from sklearn.calibration import CalibratedClassifierCV

cal_gb = CalibratedClassifierCV(best_gb, method='isotonic', cv=5)
cal_gb.fit(X_train, y_train)
```

---

## Evaluation Results

### Final Model Performance (Test Set)

| Metric | Value |
|--------|-------|
| **Recall** | ~0.83 |
| **Precision** | ~0.66 |
| **ROC-AUC** | ~0.78 |

### Confusion Matrix Interpretation

With the optimized threshold (0.36):
- **True Positives:** High identification of actual CVD cases
- **False Negatives:** Minimized (critical for healthcare)
- **False Positives:** Acceptable trade-off for higher sensitivity

---

## Decision Threshold Strategy

### Why Not Use Default 0.5?

The default probability threshold of 0.5 optimizes for accuracy, but in medical screening:
- **Missing a sick patient (FN)** → Delayed treatment, worse outcomes
- **Extra screening for healthy patient (FP)** → Additional tests, minor inconvenience

### Threshold Selection Process

Evaluated recall across thresholds from 0.10 to 0.90:

```python
thresholds = np.arange(0.1, 0.9, 0.05)
for t in thresholds:
    preds = (probabilities >= t).astype(int)
    recalls.append(recall_score(y_test, preds))
```

**Selected Threshold: 0.36**
- Achieves ~83% recall (catches most CVD cases)
- Maintains ~66% precision (acceptable false positive rate)
- Balances sensitivity with practical usability

---

## Streamlit Application

### Features

- **User-Friendly Interface:** Dropdown menus for categorical inputs
- **Automatic BMI Calculation:** Enter height and weight, BMI computed automatically
- **Probability Score:** Shows exact risk probability (0.00 - 1.00)
- **Risk Classification:** High/Low risk based on 0.36 threshold
- **Transparency:** Displays threshold and model optimization strategy
- Risk Bands (Low / Moderate / High) for interpretability, separate from screening decision


### Input Fields

| Input | Type | Range |
|-------|------|-------|
| Age | Number | Years |
| Gender | Dropdown | Male/Female |
| Height | Number | 50-250 cm |
| Weight | Number | 20-300 kg |
| Systolic BP | Number | mmHg |
| Diastolic BP | Number | mmHg |
| Cholesterol | Dropdown | Normal/Above Normal/Well Above Normal |
| Glucose | Dropdown | Normal/Above Normal/Well Above Normal |
| Smoker | Dropdown | Yes/No |
| Alcohol | Dropdown | Yes/No |
| Physical Activity | Dropdown | Yes/No |

---

## How to Run Locally

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cardio-disease-ml.git
cd cardio-disease-ml

# Install dependencies
pip install -r requirements.txt
```

### Train the Model (if not already trained)

Run the modeling notebook or training script to generate model artifacts:

```bash
# Option 1: Run Jupyter notebooks
jupyter notebook notebooks/04_modeling_and_evaluation.ipynb
```

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Deployment

### Streamlit Cloud

1. Push your repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `app.py`
5. Deploy

**Required files for deployment:**
- `app.py`
- `requirements.txt`
- `models/final_model.pkl`
- `models/scaler.pkl`



## Project Structure

```
cardio-disease-ml/
├── app.py                    # Streamlit web application
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/
│   ├── raw/
│   │   └── cardio_train.csv  # Original dataset
│   └── processed/
│       └── cardio_processed.csv  # Cleaned dataset
├── models/
│   └── final_model.pkl       # Trained model bundle  
├── notebooks/
│   ├── 01_business_and_planning.ipynb
│   ├── 02_data_validation_and_eda.ipynb
│   ├── 03_preprocessing_and_features.ipynb
│   └── 04_modeling_and_evaluation.ipynb
├── src/
│    └── model_training.py     # Model training utilities
└── deployment/
    ├── predict.py            # Prediction utilities
    ├── monitoring.py         # Conceptual monitoring plan (not automated)
    └── retraining.py         # Planned retraining strategy (not automated)
```

---

## Limitations

⚠️ **This is a decision-support prototype, NOT a medical diagnosis system.**

- **Not FDA Approved:** This tool has not undergone regulatory approval for clinical use
- **Dataset Limitations:** Model trained on a specific demographic; may not generalize to all populations
- **Feature Limitations:** Does not include all clinically relevant factors (family history, ECG, biomarkers)
- **No Longitudinal Data:** Predictions are point-in-time, not tracking disease progression
- **Threshold Trade-off:** High recall comes at the cost of more false positives
- **Requires Valid Input:** Garbage in, garbage out — unrealistic values will produce unreliable predictions

**Intended Use:**
- Educational and demonstration purposes
- Initial screening to flag patients for further evaluation
- Decision support for healthcare professionals (not replacement)

---

## Future Improvements

- [ ] Add SHAP explanations for individual predictions
- [ ] Implement confidence intervals for probability scores
- [ ] Include additional clinical features (if data available)
- [ ] Add batch prediction capability for CSV uploads
- [ ] Implement model monitoring and drift detection
- [ ] A/B testing framework for threshold optimization
- [ ] Multi-language support for broader accessibility


---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** - Machine learning
- **pandas / numpy** - Data processing
- **Streamlit** - Web application
- **matplotlib / seaborn** - Visualization


---

## License

This project is for educational and demonstration purposes.

---

## Acknowledgments

- Dataset: [Kaggle Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- Inspiration: Healthcare AI best practices for high-stakes prediction

---

**Built with ❤️ for better healthcare outcomes**
