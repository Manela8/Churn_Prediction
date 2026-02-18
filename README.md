## 📊 Customer Churn Prediction

This project predicts whether a customer will churn based on their input information. It demonstrates the full ML workflow including preprocessing, model training, and deployment. Interpretability with SHAP makes the model transparent, helps to explain predictions to stakeholders.


## 📌 Overview
This project implements an end-to-end **machine learning workflow**, including data cleaning, exploratory data analysis (EDA), model training, evaluation, and deployment.

Multiple machine learning models are trained and compared, with the best-performing model saved for inference and deployment.

---

## 📂 Project Structure
```
project/
├── data/
│ ├── raw_dataset.csv
│ └── cleaned_dataset.csv
│
├── notebooks/
│ ├── data_cleaning.ipynb
│ └── eda.ipynb
│
├── models/
│ ├── best_model.joblib
│ ├── feature_columns.joblib
│ ├── gradient_boosting_best_model.joblib
│ ├── log_reg_best_model.joblib
│ ├── random_forest_best_model.joblib
│ └── svc_best_model.joblib
│
├── src/
│ ├── init.py
│ ├── config.py
│ ├── preprocessing.py
│ ├── training.py
│ └── deployment.py
│
├── app.py
├── requirements.txt
└── README.md

```
---

## 🧪 Notebooks

### `data_cleaning.ipynb`
- Loads raw dataset
- Handles missing values
- Performs feature engineering
- Saves cleaned data to `data/cleaned_dataset.csv`

### `eda.ipynb`
- Exploratory Data Analysis
- Visualizations and statistical insights
- Identifies patterns and relationships in data

---

## 🤖 Models
The `models/` directory contains trained machine learning models saved using **joblib**:

- Logistic Regression
- Random Forest
- Support Vector Classifier (SVC)
- Gradient Boosting
- Best-performing model
- Feature columns used during training

These models are used for evaluation and deployment.

---

## 🧠 Source Code (`src/`)

- **`config.py`**
  - Stores configuration variables and constants

- **`preprocessing.py`**
  - Data preprocessing and feature transformation logic

- **`training.py`**
  - Model training, evaluation, and comparison

- **`deployment.py`**
  - Model loading and prediction utilities

---

Ensure trained model files exist in the models/ directory before running the application.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Joblib
- SHAP

 
## 🚀 Deployment

The Customer Churn Prediction model is deployed as an interactive web app using **Streamlit**. 

- 📝 Input customer details to predict churn in real-time

-⚡ Preprocessing pipelines handle both numerical (Imputer + Scaler) and categorical (Imputer + One-Hot Encoding) features

