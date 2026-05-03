# 🧠 Student Depression Prediction

> An end-to-end machine learning web application that predicts the likelihood of depression in students based on lifestyle and academic factors — with real-time explainability powered by SHAP.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Architecture](#model-architecture)
- [Explainability (SHAP)](#explainability-shap)
- [Web Application](#web-application)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

---

## 🔍 Overview

Mental health among students is a growing global concern. This project builds a **binary classification model** to predict whether a student is at risk of depression, using features such as sleep duration, study hours, CGPA, physical activity, social media usage, and stress levels.

What makes this project stand out is its **transparency** — instead of just outputting a prediction, the app uses **SHAP (SHapley Additive exPlanations)** to visually explain *why* a student is classified as high-risk or low-risk, making it interpretable and trustworthy.

**Key highlights:**
- Trained on a large dataset of **100,000 student records**
- Handles **class imbalance** using SMOTE oversampling
- Compares multiple classifiers and selects the best via **GridSearchCV**
- Interactive **Streamlit web app** with SHAP-based feature impact charts
- Fully serialized model pipeline using `joblib` for production deployment

---

## 🚀 Live Demo

> Run the app locally by following the [Installation & Setup](#installation--setup) section below.

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
student-depression/
│
├── data/
│   └── student_lifestyle_100k.csv    # Raw dataset (100k student records)
│
├── models/
│   ├── depression_model.pkl          # Trained best model (GradientBoosting/RF)
│   ├── scaler.pkl                    # StandardScaler for feature normalization
│   ├── gender_encoder.pkl            # LabelEncoder for Gender column
│   ├── department_encoder.pkl        # LabelEncoder for Department column
│   └── feature_names.pkl             # Ordered list of input features
│
├── notebooks/
│   └── EDA.ipynb                     # Full EDA, preprocessing, training & evaluation
│
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 📊 Dataset

**File:** `data/student_lifestyle_100k.csv`  
**Size:** 100,000 student records  
**Target Column:** `Depression` (Boolean — True / False)

### Features

| Feature | Type | Description |
|---|---|---|
| `Student_ID` | Integer | Unique student identifier (dropped during training) |
| `Age` | Integer | Student's age (16–40) |
| `Gender` | Categorical | Male / Female |
| `Department` | Categorical | Academic department (e.g., Science, Engineering, Medical) |
| `CGPA` | Float | Cumulative Grade Point Average (0.0–10.0) |
| `Sleep_Duration` | Float | Average sleep per night in hours |
| `Study_Hours` | Float | Daily study hours |
| `Social_Media_Hours` | Float | Daily social media usage in hours |
| `Physical_Activity` | Integer | Weekly physical activity in minutes |
| `Stress_Level` | Integer | Self-reported stress level (1–10) |
| `Depression` | Boolean | **Target** — True if depressed, False otherwise |

---

## ⚙️ ML Pipeline

The full machine learning pipeline is documented in `notebooks/EDA.ipynb` and covers 16 well-defined stages:

### Step 1 — Data Loading
Raw CSV loaded into a Pandas DataFrame and visually inspected using `.head()`, `.info()`, and `.describe()`.

### Step 2 — Exploratory Data Analysis (EDA)
- Distribution plots for all numeric features
- Gender distribution via countplot
- Depression class balance check
- Correlation heatmap to detect multicollinearity

### Step 3 — Target Analysis
Relationship between each feature and the `Depression` target is visualized to understand predictive power before modeling.

### Step 4 — Preprocessing
- `Student_ID` dropped (non-informative identifier)
- `Gender` and `Department` encoded using `LabelEncoder`
- All features standardized using `StandardScaler`

### Step 5 — Class Imbalance Handling
The dataset had imbalanced depression labels. **SMOTE (Synthetic Minority Oversampling Technique)** was applied on the training set to synthetically balance classes, preventing the model from being biased toward the majority class.

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

### Step 6 — Train/Test Split
An 80/20 stratified split was used to preserve the class distribution in both subsets.

### Step 7–9 — Model Training
Three classifiers were trained and compared:

| Model | Notes |
|---|---|
| `LogisticRegression` | Baseline linear model with `class_weight='balanced'` |
| `GradientBoostingClassifier` | Ensemble boosting model |
| `RandomForestClassifier` | Ensemble bagging model — **selected as best** |

### Step 10 — Hyperparameter Tuning
`GridSearchCV` with cross-validation was applied to find the optimal hyperparameters for the best-performing model.

### Step 11 — Evaluation
Models were evaluated using:
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix**
- **ROC-AUC Score**

### Step 12 — Feature Importance
Feature importances were extracted from the final tree-based model and visualized as a bar chart to understand which factors contribute most to depression prediction.

### Step 13 — Model Serialization
All artifacts were saved using `joblib` for reuse in the Streamlit app:

```python
joblib.dump(best_model,    "models/depression_model.pkl")
joblib.dump(scaler,        "models/scaler.pkl")
joblib.dump(le_gender,     "models/gender_encoder.pkl")
joblib.dump(le_dept,       "models/department_encoder.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")
```

---

## 🤖 Model Architecture

The final production model is a **Gradient Boosting / Random Forest Classifier** selected through `GridSearchCV`. Here's why tree-based ensembles were chosen:

- **No assumptions about data distribution** — unlike Logistic Regression
- **Handles non-linear relationships** between lifestyle factors and depression
- **Naturally provides feature importances** for interpretability
- **Robust to outliers** in self-reported metrics like stress level or social media hours
- **Compatible with SHAP's TreeExplainer** for fast, exact explanations

---

## 💡 Explainability (SHAP)

One of the most important aspects of this project is **model transparency**. After each prediction, the app uses `shap.TreeExplainer` to compute SHAP values — a game-theory-based method that assigns each feature a contribution score for *that specific prediction*.

### How it works

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_scaled)
```

- A **positive SHAP value** means the feature *pushed the prediction toward depression*
- A **negative SHAP value** means the feature *pulled the prediction away from depression*

### What the app shows

- 📊 **Bar chart** of all feature SHAP values (red = increases risk, green = decreases risk)
- 📋 **Detailed table** of feature contributions (expandable)
- 🔺🔻 **Top 3 key factors** that influenced the specific prediction, in plain language

---

## 🌐 Web Application

The Streamlit app (`app.py`) provides a clean, interactive interface:

### Inputs
| Input | Widget | Range |
|---|---|---|
| Age | Number input | 16–40 |
| Gender | Selectbox | Male / Female |
| Department | Selectbox | All available departments |
| CGPA | Number input | 0.0–10.0 |
| Sleep Duration | Number input | 0–12 hours |
| Study Hours | Number input | 0–12 hours |
| Social Media Hours | Number input | 0–10 hours |
| Physical Activity | Number input | 0–300 min/week |
| Stress Level | Slider | 1–10 |

### Outputs
- **Depression probability** as a percentage
- **Risk classification**: High Risk (> 40%) or Low Risk
- **SHAP bar chart** with color-coded feature impacts
- **Top 3 influential factors** with directional indicators

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+ recommended
- pip package manager

### 1. Clone the repository

```bash
git clone https://github.com/jidnyasadthakre07/student-depression-prediction.git
cd student-depression-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (if models/ folder is not present)

Open and run all cells in `notebooks/EDA.ipynb`. This will generate all `.pkl` files inside the `models/` directory.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage

1. Launch the app using `streamlit run app.py`
2. Fill in the student's profile using the input fields on the left
3. Click the **"Predict"** button
4. View the **depression probability**, **risk level**, and **SHAP explanation chart**
5. Expand the **"View detailed feature contributions"** section for the full breakdown

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.13 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, imbalanced-learn |
| **Explainability** | SHAP |
| **Web App** | Streamlit |
| **Serialization** | Joblib |
| **Notebook** | Jupyter Notebook |

---

## 📈 Results

Three models were trained, evaluated, and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline — lower recall on minority class |
| Gradient Boosting | Strong precision, moderate recall |
| **Random Forest (Best)** | Best overall F1-score, selected via GridSearchCV |

> Exact metric values (accuracy, precision, recall, F1, ROC-AUC) are available in the classification reports inside `notebooks/EDA.ipynb`.

**Key insight from feature importance analysis:**
> Stress Level, Sleep Duration, and Physical Activity consistently ranked as the top predictors of depression risk across all tree-based models.

---

## 🔮 Future Improvements

- [ ] Add ROC curve and confusion matrix visualization in the Streamlit app
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces for public access
- [ ] Add SHAP waterfall plots for richer per-prediction explanations
- [ ] Collect real survey data to replace synthetic dataset
- [ ] Add multi-language support for broader accessibility
- [ ] Implement user session history to track predictions over time
- [ ] Add a counselor recommendation module based on risk level

---
