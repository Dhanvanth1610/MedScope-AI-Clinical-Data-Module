<div align="center">

# 🩺 MedScope AI — Clinical Data Module

### Explainable heart-disease and stroke-risk screening, served through a FastAPI backend and a clean web dashboard

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?logo=scikit-learn&logoColor=white)
![Frontend](https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JS-E34F26?logo=html5&logoColor=white)
![Tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-prototype-blue)

> ⚠️ **Medical disclaimer:** This is an AI-assisted **screening prototype**. It does **not** provide a medical diagnosis and must not replace professional clinical judgement.

</div>

---

## 📖 Overview

**MedScope AI** is designed as a unified healthcare AI platform with three modules: **ECG Analysis**, **X-Ray Analysis** and **Clinical Data**. This repository contains the **Clinical Data Module**.

It takes structured patient data (vitals, lab values, history and lifestyle) and returns:

- a **risk prediction** (lower-risk vs. higher-risk pattern),
- a **probability score**,
- the **top 4 factors** that influenced that prediction, in plain language,
- a **safety disclaimer** with every response.

Two independent models are exposed through one API:

| Model | Question it answers | Dataset |
|---|---|---|
| ❤️ **Heart Disease** | Does this clinical profile show a higher-risk pattern for heart disease? | `heart.csv`, 1,025 records, 13 features |
| 🧠 **Stroke Risk** | Does this profile show an elevated risk pattern for stroke? | `stroke_prediction.csv`, 5,110 records, 10 features |

---

## ✨ Key Features

- **Two production-style ML pipelines** — preprocessing and model are bundled into one scikit-learn `Pipeline`, so training and inference use identical transforms.
- **Per-prediction explainability** — each response lists the 4 most influential features and whether each one *increases* or *lowers* the model's risk score.
- **Leakage-aware evaluation** — the heart dataset contains heavy duplication, so it is evaluated with `StratifiedGroupKFold`, with an assertion that no group appears in both train and validation folds.
- **Imbalance-aware stroke modelling** — only ~4.9% of stroke records are positive, so models use class balancing and are judged on recall, PR-AUC and ROC-AUC rather than accuracy.
- **Strict input validation** — Pydantic schemas enforce clinical value ranges and allowed categories; bad requests get a clean `422` with readable details.
- **Safe error handling** — unexpected failures return a masked `500`, so no stack traces leak.
- **Missing-value tolerance** — `bmi` is optional for stroke prediction and is median-imputed inside the pipeline.
- **Interactive dashboard** — switch between heart and stroke assessments, load preset patients, and see results with feature explanations.
- **Live service monitor** — the UI shows whether the API and both models are online.
- **Automated tests** — 6 API tests covering health, valid predictions, missing BMI and invalid inputs.

---

## 🏗️ System Architecture

The Clinical Data Module is one branch of the wider MedScope AI platform. Alongside the ECG and X-Ray modules, it feeds a shared screening output that is shown to a healthcare user or professional.

<div align="center">
  <img src="docs/MedScope%20AI%20-%20System%20Architecture.png" alt="MedScope AI System Architecture" width="520"/>
</div>

**Clinical module data flow:**

```
Clinical Patient Data → Data Preprocessing → Heart & Stroke ML Models → Prediction + Explainability → Clinical API → Screening Output
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **ML** | scikit-learn (Logistic Regression, Random Forest, HistGradientBoosting), pandas, NumPy, joblib |
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **Frontend** | Vanilla HTML, CSS and JavaScript (no build step) |
| **Testing** | `unittest` with FastAPI `TestClient` |

---

## 📁 Project Structure

```
clinical-module/
├── api/
│   └── main.py                  # FastAPI app: endpoints, validation, error handling
├── src/
│   ├── preprocessing.py         # Feature lists, grouping, scaler / imputer / encoder pipelines
│   ├── train_heart.py           # Heart model training + grouped cross-validation
│   ├── train_stroke.py          # Stroke model training + stratified CV, imbalance handling
│   ├── predict.py               # Cached model loading + inference functions
│   └── explain.py               # Feature-contribution explanations
├── models/
│   ├── heart_pipeline.pkl       # Trained heart pipeline
│   └── stroke_pipeline.pkl      # Trained stroke pipeline
├── data/
│   ├── heart.csv
│   └── stroke_prediction.csv
├── reports/                     # Saved evaluation metrics (JSON)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   └── test_api.py
└── docs/
    └── MedScope AI - System Architecture.png
```

---

## 🧠 How It Works

### 1. Preprocessing
- **Heart:** `StandardScaler` on all 13 numeric clinical features.
- **Stroke:** a `ColumnTransformer` that applies
  - median imputation + scaling to `age`, `avg_glucose_level`, `bmi`, `hypertension`, `heart_disease`
  - one-hot encoding (`handle_unknown='ignore'`) to `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`

All of this lives *inside* the pipeline, so it is fitted only on training folds and cannot leak validation data.

### 2. Model selection
Candidate models are compared with cross-validated out-of-fold predictions, and the best is retrained on the full dataset and saved.

| Module | Candidates compared | Selected |
|---|---|---|
| Heart | Logistic Regression, Random Forest | **Logistic Regression** |
| Stroke | Logistic Regression, Random Forest, HistGradientBoosting (all class-balanced) | **Logistic Regression (Balanced)** |

### 3. Explainability
Both selected models are linear, so each feature's contribution is `scaled value × learned coefficient`. The top 4 by absolute size are returned with a direction: *increases* or *lowers model risk score*. For stroke, one-hot columns are summed back to their original feature (for example, all `work_type_*` columns become one **Work Type** contribution).

> These explanations describe **what influenced the model**, not medical causation.

---

## 📊 Model Performance

All numbers are **out-of-fold** results from 5-fold cross-validation, taken from `reports/`.

### ❤️ Heart Disease: Logistic Regression
*StratifiedGroupKFold, grouped by feature hash*

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|:---:|:---:|:---:|:---:|:---:|
| **83.6%** | **81.1%** | **88.8%** | **84.8%** | **0.899** |

Confusion matrix (rows = actual, columns = predicted):

|  | Pred. 0 | Pred. 1 |
|---|:---:|:---:|
| **Actual 0** | 390 | 109 |
| **Actual 1** | 59 | 467 |

**Data-quality note:** the raw file has 1,025 rows but only **302 unique feature combinations** (723 duplicates). A plain random split would put copies of the same patient in both train and validation sets and inflate the scores, so grouped cross-validation is used to keep the estimate honest.

### 🧠 Stroke Risk: Logistic Regression (Balanced)
*StratifiedKFold, 5,109 records after removing one `gender = Other` row; 4.87% positive class*

| Recall | Precision | F1 | ROC-AUC | PR-AUC | Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **79.1%** | 13.3% | 22.8% | **0.837** | 0.188 | 73.9% |

Confusion matrix:

|  | Pred. 0 | Pred. 1 |
|---|:---:|:---:|
| **Actual 0** | 3,581 | 1,279 |
| **Actual 1** | 52 | 197 |

**Why these numbers look the way they do:** with roughly 1 stroke per 20 patients, the model is tuned to **catch most true stroke cases (79% recall)** at the cost of many false alarms (13% precision). That is a deliberate screening trade-off, since missing a stroke is costlier than flagging an extra patient for follow-up. This is also why accuracy is treated as a secondary metric here.

<details>
<summary><b>Stroke model comparison</b></summary>

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Balanced)** ✅ | 0.791 | 0.133 | 0.228 | 0.837 | 0.188 |
| Random Forest (Balanced) | 0.755 | 0.130 | 0.222 | 0.821 | 0.164 |
| HistGradientBoosting (Balanced) | 0.518 | 0.167 | 0.253 | 0.821 | 0.175 |

</details>

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 (3.10+ should also work)
- `pip`

### 1. Clone and set up

```bash
git clone <your-repo-url>
cd clinical-module

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install fastapi "uvicorn[standard]" pydantic scikit-learn pandas numpy joblib httpx
```

> Pre-trained models are included in `models/`, so you can skip straight to step 3. If you retrain, use the same scikit-learn version to avoid pickle warnings.

### 2. (Optional) Retrain the models

```bash
python -m src.train_heart
python -m src.train_stroke
```

This regenerates `models/*.pkl` and the JSON metrics in `reports/`.

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

- API root: `http://127.0.0.1:8000`
- Interactive docs (Swagger): `http://127.0.0.1:8000/docs`

### 4. Open the dashboard

In a second terminal:

```bash
cd frontend
python -m http.server 5500
```

Then visit **http://127.0.0.1:5500**. The frontend expects the API at `http://127.0.0.1:8000` (set by `API_BASE_URL` in `frontend/app.js`).

### 5. Run the tests

```bash
python -m unittest discover tests
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status and model-load state |
| `POST` | `/predict/heart` | Heart-disease risk prediction |
| `POST` | `/predict/stroke` | Stroke risk prediction |

### `POST /predict/heart`

**Request**

```json
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
  "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
```

**Response**

```json
{
  "module": "heart",
  "prediction": 0,
  "probability": 0.2099,
  "important_features": [
    {
      "feature": "Major Vessels Count",
      "value": "2.0",
      "effect": "lowers model risk score",
      "description": "Major Vessels Count (2.0) lowers model risk score"
    }
  ],
  "message": "The model detected a lower-risk pattern in the provided clinical data.",
  "disclaimer": "This is an AI-assisted screening prototype and does not provide a medical diagnosis."
}
```
*(`important_features` is shortened here; the API returns the top 4.)*

<details>
<summary><b>Heart input fields and valid ranges</b></summary>

| Field | Meaning | Range / values |
|---|---|---|
| `age` | Age in years | 1–120 |
| `sex` | 1 = male, 0 = female | 0–1 |
| `cp` | Chest pain type | 0–3 |
| `trestbps` | Resting blood pressure (mm Hg) | 50–300 |
| `chol` | Serum cholesterol (mg/dl) | 80–700 |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0–1 |
| `restecg` | Resting ECG result | 0–2 |
| `thalach` | Maximum heart rate achieved | 40–250 |
| `exang` | Exercise-induced angina | 0–1 |
| `oldpeak` | ST depression | 0.0–10.0 |
| `slope` | Slope of peak exercise ST segment | 0–2 |
| `ca` | Major vessels count | 0–4 |
| `thal` | Thalassemia result | 0–3 |

</details>

### `POST /predict/stroke`

**Request**

```json
{
  "gender": "Male",
  "age": 67.0,
  "hypertension": 0,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

**Response**

```json
{
  "module": "stroke",
  "prediction": 1,
  "probability": 0.8321,
  "important_features": [
    {
      "feature": "Age",
      "value": "67.0",
      "effect": "increases model risk score",
      "description": "Age (67.0) increases model risk score"
    }
  ],
  "message": "The model detected an elevated risk score pattern for stroke screening.",
  "disclaimer": "This is an AI-assisted screening prototype and does not provide a medical diagnosis."
}
```

<details>
<summary><b>Stroke input fields and valid values</b></summary>

| Field | Allowed values |
|---|---|
| `gender` | `Male`, `Female`, `Other` |
| `age` | 0–120 |
| `hypertension` | 0 or 1 |
| `heart_disease` | 0 or 1 |
| `ever_married` | `Yes`, `No` |
| `work_type` | `Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked` |
| `Residence_type` | `Urban`, `Rural` |
| `avg_glucose_level` | 40–350 |
| `bmi` | 10–100 *(optional; median-imputed if omitted)* |
| `smoking_status` | `formerly smoked`, `never smoked`, `smokes`, `Unknown` |

</details>

### Error responses

Invalid input returns `422` with readable details instead of a stack trace:

```json
{
  "error": "Validation Error",
  "details": ["body -> chol: Input should be greater than or equal to 80"],
  "message": "Provided input failed schema validation. Please check field types and allowed ranges."
}
```

### Quick test with cURL

```bash
curl -X POST http://127.0.0.1:8000/predict/heart \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

---

## 🛡️ Responsible AI & Safety

- **Not a diagnostic tool.** Every response carries a disclaimer, and the wording is deliberately cautious ("higher-risk pattern", "elevated risk score pattern").
- **Explanations ≠ causation.** Feature effects describe the model's behaviour, not medical cause.
- **Public datasets only.** Models are trained on public Kaggle-style datasets and have not been validated on real clinical populations.
- **Known limitations:**
  - The heart dataset is small (302 unique patient profiles), so performance may not generalise.
  - The stroke model has low precision by design and will produce many false positives.
  - CORS is open (`*`) for local development; restrict it before any real deployment.
  - There is no authentication or audit logging.

---

## 🗺️ Roadmap

- [ ] Integrate with the **ECG** and **X-Ray** modules into the full MedScope AI platform
- [ ] Add calibrated probabilities and decision-threshold tuning
- [ ] Add SHAP-based explanations for non-linear models
- [ ] Containerise with Docker and add CI for the test suite
- [ ] Add authentication, rate limiting and request logging
- [ ] Validate on external datasets

---

## 👤 Author

**Ashwath L**

---

<div align="center">

**MedScope AI** · Clinical Data Module · *AI-assisted screening, not a medical diagnosis.*

</div>
