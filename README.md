# 🛡️ ChurnGuard AI — Customer Churn Prediction + Retention Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-FF6600?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge)
![Gemini AI](https://img.shields.io/badge/Gemini%202.5%20Flash-AI%20Strategy-4285F4?style=for-the-badge&logo=google&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

**End-to-end MLOps system: predicts customer churn AND generates AI-powered retention strategies**

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [📊 Results](#-real-model-results) · [🤖 AI Layer](#-ai-recommendations-gemini-25-flash) · [🐳 Deploy](#-deployment)

</div>

---

## 📌 The Problem

Telecom companies lose **15–25% of customers per year**. Most ML projects stop at predicting *who* will churn. That's only half the job.

**ChurnGuard AI does the full job:**

| Standard ML Project | ChurnGuard AI |
|---|---|
| Predicts churn: Yes/No | Probability + risk level per customer |
| No explanation | Flags exact risk drivers per customer |
| No action | AI retention strategy (Gemini 2.5 Flash) |
| Jupyter notebook | FastAPI + Streamlit production app |
| No tracking | MLflow experiment tracking + model registry |
| No reproducibility | DVC data versioning pipeline |
| Manual deploy | Docker + GitHub Actions CI/CD + Render |

---

## 🏗️ Architecture

```
Customer Data
      │
      ▼
Streamlit UI ──────► FastAPI Backend
                           │
               ┌───────────┴────────────┐
               │                        │
         XGBoost Model           Gemini 2.5 Flash
         (Optuna tuned)         (Google AI Studio)
               │                        │
               └───────────┬────────────┘
                           │
                  ┌────────▼────────┐
                  │ Unified Response │
                  │  churn_prob     │
                  │  risk_level     │
                  │  ai_strategy    │
                  └─────────────────┘

MLOps Infrastructure:
  MLflow ──── Experiment tracking + model registry
  DVC ──────── Data pipeline versioning
  Docker ────── Containerized services (API + UI + MLflow)
  GitHub Actions ── CI/CD: lint → test → build → deploy
  Render ────── Cloud deployment
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/haseeb774/Churn-guard-ML-project-with-mlops-and-AI-capability.git
cd Churn-guard-ML-project-with-mlops-and-AI-capability
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set GOOGLE_AI_STUDIO_API_KEY
```

Get your free API key at [aistudio.google.com](https://aistudio.google.com/apikey)

### 3. Add dataset

Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) → place at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 4. Train model

```bash
python run_pipeline.py --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Outputs:
- `outputs/model.pkl` — XGBoost model (Optuna tuned)
- `outputs/analysis/` — 8 EDA + evaluation plots
- MLflow run logged at `http://localhost:5000`

### 5. Launch

```bash
# Terminal 1
uvicorn app:app --reload

# Terminal 2
streamlit run streamlit_app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## 📊 Real Model Results

XGBoost + Optuna (30 trials). These are the actual numbers from this repo's training run.

### Confusion Matrix

![Confusion Matrix](outputs/analysis/06_confusion_matrix.png)

| | Predicted No Churn | Predicted Churn |
|---|---|---|
| **Actual No Churn** | 857 ✅ | 176 ❌ |
| **Actual Churn** | 120 ❌ | 254 ✅ |

### ROC Curve — AUC: 0.839

![ROC Curve](outputs/analysis/07_roc_curve.png)

### Model Metrics

| Metric | Score |
|---|---|
| **ROC-AUC** | **0.839** |
| Accuracy | 81.2% |
| Precision (Churn) | 59.1% |
| Recall (Churn) | 67.9% |
| F1 Score (Churn) | 63.2% |

### Top 15 Feature Importances

![Feature Importance](outputs/analysis/08_feature_importance.png)

| Rank | Feature | Importance | Business Meaning |
|---|---|---|---|
| 1 | **Contract** | 0.334 | Month-to-month = 42.7% churn rate |
| 2 | **InternetService** | 0.125 | Fiber optic = highest churn segment |
| 3 | **PaymentMethod_Electronic check** | 0.083 | Highest churn payment method |
| 4 | **OnlineSecurity** | 0.055 | Absent = 31.4% churn vs 14.6% with it |
| 5 | **tenure** | 0.050 | Short tenure = highest churn risk |
| 6 | **PaperlessBilling** | 0.038 | Correlated with higher monthly charges |
| 7 | **StreamingMovies** | 0.030 | Service stickiness signal |

---

## 📈 EDA Highlights

### Churn Distribution

![Churn Distribution](outputs/analysis/01_churn_distribution.png)

26.6% churn rate (1,869 of 7,032 customers) — significant class imbalance handled via XGBoost `scale_pos_weight`.

### Churn by Contract Type

![Churn by Contract](outputs/analysis/02_churn_by_contract.png)

**Month-to-month: 42.7% churn** vs 2.8% for two-year contracts. Contract type is the single strongest predictor.

### Tenure Distribution

![Tenure by Churn](outputs/analysis/03_tenure_by_churn.png)

Churners are heavily concentrated in months 0–12. Customers who survive the first year are significantly more likely to stay long-term.

### Monthly Charges

![Monthly Charges](outputs/analysis/04_monthly_charges.png)

Churned customers pay **~$20/month more** on average — they're paying premium prices but not getting enough perceived value.

### Senior Citizen Churn

![Senior Churn](outputs/analysis/05_senior_churn.png)

Senior citizens churn at **41.7%** vs 23.7% for non-seniors — nearly 2× the rate.

---

## 🤖 AI Recommendations — Gemini 2.5 Flash

After ML prediction, `/predict-with-recommendations` calls **Gemini 2.5 Flash** to generate a structured, customer-specific retention strategy.

### Example Response — High-Risk Customer

**Input:** Month-to-month · 3mo tenure · $87/mo · Electronic check · No security · No tech support

**ML:** Churn probability 84.7% · Risk: HIGH

**Gemini output:**
```
1. ROOT CAUSE ANALYSIS
   This customer displays three compounding churn signals: month-to-month 
   contract (zero switching friction), 3-month tenure (trial mindset, not 
   yet committed), and electronic check payment — the highest-churn method 
   in this dataset. At $87/month with no security or tech support, the 
   value-to-cost ratio is insufficient to motivate loyalty.

2. IMMEDIATE ACTIONS (next 7 days)
   1. Outbound call within 48 hours — before they start comparing competitors
   2. Offer free 30-day trial: Online Security + Tech Support bundle
   3. Present 1-year contract at 20% discount ($69.60/mo vs $87.00)

3. TAILORED RETENTION OFFER
   Switch to a 1-year contract: $69.60/month (save $208.80/year), Online 
   Security free for 6 months, dedicated priority support line. Limited 
   to customers contacted this week.

4. LONG-TERM STRATEGY (6 months)
   1. Migrate to automatic bank transfer payment — reduces churn ~40%
   2. 90-day check-in call at month 6 to cross-sell streaming bundle

5. RISK SCORE INTERPRETATION
   84.7% means approximately 85 of 100 customers with this exact profile 
   will cancel within the next quarter without targeted intervention.
```

### API Call

```python
import requests

payload = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 87.0, "TotalCharges": 261.0
}

res = requests.post(
    "http://localhost:8000/predict-with-recommendations",
    json=payload
)
print(res.json())
```

---

## 🐳 Deployment

### Option A — Docker Compose (full stack)

```bash
# Copy env vars
cp .env.example .env
# Edit .env with your GOOGLE_AI_STUDIO_API_KEY

# Launch API + UI + MLflow together
docker compose up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| UI | http://localhost:8501 |
| MLflow | http://localhost:5000 |

### Option B — Docker (API only)

```bash
docker build -t churnguard-ai .

docker run -p 8000:8000 \
  -e GOOGLE_AI_STUDIO_API_KEY=$GOOGLE_AI_STUDIO_API_KEY \
  churnguard-ai
```

### Option C — Render (free cloud deploy)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect this repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Add env var: `GOOGLE_AI_STUDIO_API_KEY`
7. Deploy — free tier works fine for the API

Add `RENDER_DEPLOY_HOOK_URL` as a GitHub secret to trigger auto-deploys on every push to `main`.

---

## ⚙️ CI/CD Pipeline — GitHub Actions

Triggers on every push to `main` or `develop`:

```
push to main
     │
     ├── 🔍 lint       ruff — code quality check
     ├── 🧪 test       pytest — unit tests
     ├── 🐳 build      docker build — image validation
     └── 🚀 deploy     render deploy hook (main branch only)
```

View pipeline: `Actions` tab on GitHub.

---

## 🔬 MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Open http://localhost:5000
```

Every training run logs:
- All Optuna hyperparameters
- Accuracy, Precision, Recall, F1, ROC-AUC
- XGBoost model artifact
- Run name with timestamp

---

## 📁 Project Structure

```
Churn-guard-ML-project-with-mlops-and-AI-capability/
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml               # CI/CD pipeline
│
├── src/                            # Modular source code
│   ├── __init__.py
│   ├── logging.py                  # Fixed logging (dir vs file bug resolved)
│   ├── exception.py                # Custom exception handler
│   ├── data_ingest.py              # Data loading + raw save
│   ├── data_transform.py           # Feature engineering + encoding
│   └── model_train.py              # XGBoost + Optuna + MLflow + EDA plots
│
├── Notebook/                       # Jupyter EDA notebooks
│
├── data/
│   ├── raw/                        # DVC tracked
│   └── processed/                  # DVC tracked
│
├── outputs/
│   ├── model.pkl                   # Trained XGBoost (Optuna tuned)
│   └── analysis/                   # 8 plots generated at training
│
├── app.py                          # FastAPI backend + Gemini AI
├── streamlit_app.py                # Streamlit UI
├── run_pipeline.py                 # End-to-end pipeline (argparse, no hardcoded paths)
├── Dockerfile                      # Multi-stage production build
├── docker-compose.yml              # Full stack: API + UI + MLflow
├── dvc.yaml                        # DVC pipeline stages
├── .gitignore                      # .env excluded — never commit secrets
├── .env.example                    # Safe template to commit
├── requirements.txt
└── README.md
```

---

## ⚠️ Security Note

**Never commit your `.env` file.** It contains your API key.

```bash
# If you accidentally committed it:
git rm --cached .env
git commit -m "remove .env from tracking"
# Then rotate your API key immediately at aistudio.google.com
```

---

## 📈 Business Impact

At Telco dataset scale (7,032 customers):

| | Value |
|---|---|
| Churners identified by model (Recall 67.9%) | ~1,270 of 1,869 |
| Revenue per churner (avg $65/mo × 12) | ~$780/year |
| Revenue at risk captured by model | ~$990K |
| Recovery rate with AI-guided retention (30%) | **~$297K/year** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost + Optuna hyperparameter tuning |
| Experiment Tracking | MLflow (SQLite backend) |
| API | FastAPI + Pydantic + Uvicorn |
| UI | Streamlit |
| AI Layer | Gemini 2.5 Flash (Google AI Studio) |
| Data Versioning | DVC |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud Deploy | Render |
| Logging | Python logging (custom formatter) |

---

## 👤 Author

**Haseeb ur Rehman** — Data Scientist · MLOps Engineer

[![GitHub](https://img.shields.io/badge/GitHub-haseeb774-181717?style=flat&logo=github)](https://github.com/haseeb774)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/haseeb-u-rehman-4822bb369)

---

<div align="center"><i>Leave a ⭐ if this helped you</i></div>