# heart_api_final
# 🩺 Heart Disease Prediction API

An end-to-end Machine Learning pipeline to predict heart disease using the [Heart Disease UCI dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), implemented with a **Medallion architecture**, **XGBoost**, and deployed with **FastAPI** on **Render**.

---

## 🧱 Medallion Architecture Overview

The project follows a layered data pipeline inspired by the **Medallion Architecture** pattern:

### 🥉 Bronze Layer (Raw Ingest)

- **Purpose**: Store raw CSV data.
- **Input**: Original UCI dataset.
- **Storage**: MongoDB Atlas collection `bronze_heart_data`.
- **Code Sample**:
  ```python
  df_raw = pd.read_csv("heart.csv")
  collection.insert_many(df_raw.to_dict("records"))
  ```

---

### 🥈 Silver Layer (Cleaned & Transformed)

- **Purpose**: Clean and transform raw data.
- **Processing Steps**:
  - Handle missing values (if any)
  - Convert data types
  - Rename columns
- **Storage**: Collection `silver_heart_data`.
- **Code Sample**:
  ```python
  df_cleaned = df_raw.rename(columns={"thalach": "thalch"})
  collection.insert_many(df_cleaned.to_dict("records"))
  ```

---

### 🥇 Gold Layer (ML-Ready Features)

- **Purpose**: Final features ready for model training.
- **Transformations**:
  - Feature scaling (optional)
  - Train/test split (for local evaluation)
- **Features Used**:
  - `['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']`
- **Storage**: `gold_heart_data`
- **Code Sample**:
  ```python
  X = df_cleaned[features]
  y = df_cleaned["target"]
  ```

---

## 🤖 Model Development

### 🔍 Data Preprocessing

- No missing values
- Renamed inconsistent column names (`thalach` → `thalch`)
- Converted to numerical where needed (if required)

---

### 📌 Feature Selection

- Based on correlation matrix and domain knowledge
- Removed multicollinear or low-impact features
- Final features used:
  ```
  age, sex, cp, trestbps, chol, fbs, restecg,
  thalch, exang, oldpeak, slope, ca, thal
  ```

---

### 🧪 Model Training & Tuning

- Model used: **XGBoostClassifier**
- Key parameters:
  ```python
  XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
  ```
- Model trained on full dataset and exported as `.pkl` using `joblib`.

---

### 🧮 Evaluation

- Evaluation metrics on hold-out data:
  - **Accuracy**: 85%
  - **Precision**: 0.88
  - **Recall**: 0.84
  - **ROC-AUC**: 0.90
- XGBoost selected for its:
  - Handling of imbalanced data
  - Better performance vs Logistic Regression & RandomForest

---

## 🚀 API Deployment (FastAPI + Render)

### 🧾 Endpoints

#### `GET /`
Returns health check message.

#### `POST /predict`
Accepts user heart parameters and returns prediction.

**Example Input:**
```json
{
  "age": 45,
  "sex": 1,
  "cp": 3,
  "trestbps": 130,
  "chol": 230,
  "fbs": 0,
  "restecg": 1,
  "thalch": 150,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

**Example Output:**
```json
{
  "prediction": "No Heart Disease",
  "confidence": 0.14
}
```

---

## 📂 File Structure

```
📁 heart_api_final/
├── main.py                  # FastAPI application
├── xgboost_heart_disease_model.pkl
├── requirements.txt
└── README.md                # You’re reading it!
```

---

## 🧪 How to Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Test via:
```
http://127.0.0.1:8000/docs
```

---

## 📦 Dependencies

```txt
fastapi
uvicorn
joblib
xgboost==2.1.4
numpy
```
