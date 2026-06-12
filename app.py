"""
app.py — ChurnGuard AI FastAPI Backend

Endpoints:
  GET  /                         Health check
  GET  /health                   Model + system status
  GET  /get_analysis_images      List EDA plot filenames
  GET  /image/{image_name}       Serve EDA plot image
  POST /predict                  ML prediction only
  POST /predict-with-recommendations  ML prediction + Gemini AI strategy
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import google.genai as genai  # Swapped from anthropic

app = FastAPI(
    title="ChurnGuard AI",
    description="Customer churn prediction + AI-powered retention strategy",
    version="2.0.0",
)

# ── Model Loading ─────────────────────────────────────────────────────────────
# Load lazily so the app starts even if model.pkl doesn't exist yet

_model = None

def get_model():
    global _model
    if _model is None:
        model_path = "outputs/model.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run: python run_pipeline.py --data <path>"
            )
        _model = joblib.load(model_path)
    return _model


ANALYSIS_PATH = "outputs/analysis"

# ── Schemas ───────────────────────────────────────────────────────────────────

class RawCustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ── Encoding (must match data_transform.py exactly) ──────────────────────────

MAPPING = {
    "gender":           {"Female": 0.0, "Male": 1.0},
    "Partner":          {"No": 0.0, "Yes": 1.0},
    "Dependents":       {"No": 0.0, "Yes": 1.0},
    "PhoneService":     {"No": 0.0, "Yes": 1.0},
    "MultipleLines":    {"No": 0.0, "Yes": 1.0, "No phone service": 0.0},
    "InternetService":  {"No": 0.0, "DSL": 1.0, "Fiber optic": 2.0},
    "OnlineSecurity":   {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "OnlineBackup":     {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "DeviceProtection": {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "TechSupport":      {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "StreamingTV":      {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "StreamingMovies":  {"No": 0.0, "Yes": 1.0, "No internet service": 0.0},
    "Contract":         {"Month-to-month": 0.0, "One year": 1.0, "Two year": 2.0},
    "PaperlessBilling": {"No": 0.0, "Yes": 1.0},
}


def encode_customer(d: dict) -> pd.DataFrame:
    processed = {}
    for key, value in d.items():
        if key in MAPPING:
            processed[key] = MAPPING[key].get(value, 0.0)
        else:
            processed[key] = value

    pm = d["PaymentMethod"]
    processed["PaymentMethod_Bank transfer (automatic)"]  = 1.0 if pm == "Bank transfer (automatic)" else 0.0
    processed["PaymentMethod_Credit card (automatic)"]    = 1.0 if pm == "Credit card (automatic)" else 0.0
    processed["PaymentMethod_Electronic check"]           = 1.0 if pm == "Electronic check" else 0.0
    processed["PaymentMethod_Mailed check"]               = 1.0 if pm == "Mailed check" else 0.0
    del processed["PaymentMethod"]

    return pd.DataFrame([processed])


def get_risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


# ── AI Prompt Builder ─────────────────────────────────────────────────────────

def build_retention_prompt(customer: dict, prob: float, prediction: int) -> str:
    risk = get_risk_level(prob)
    return f"""You are a senior customer retention strategist at a telecom company.

A machine learning model has analyzed a customer profile and returned:

CHURN PREDICTION : {"WILL CHURN" if prediction == 1 else "WILL STAY"}
CHURN PROBABILITY: {prob:.1%}
RISK LEVEL       : {risk}

CUSTOMER PROFILE:
- Tenure          : {customer['tenure']} months
- Contract        : {customer['Contract']}
- Monthly Charges : ${customer['MonthlyCharges']:.2f}
- Total Charges   : ${customer['TotalCharges']:.2f}
- Internet Service: {customer['InternetService']}
- Online Security : {customer['OnlineSecurity']}
- Tech Support    : {customer['TechSupport']}
- Payment Method  : {customer['PaymentMethod']}
- Senior Citizen  : {"Yes" if customer['SeniorCitizen'] == 1 else "No"}
- Partner         : {"Yes" if customer['Partner'] == 1 else "No"}
- Dependents      : {"Yes" if customer['Dependents'] == 1 else "No"}
- Phone Service   : {"Yes" if customer['PhoneService'] == 1 else "No"}
- Multiple Lines  : {customer['MultipleLines']}

Provide a structured retention strategy with exactly these 5 sections:

**1. ROOT CAUSE ANALYSIS**
2-3 sentences explaining WHY this customer is likely to churn based on their profile.

**2. IMMEDIATE ACTIONS** (within 7 days)
Three specific, numbered actions the retention team should take immediately.

**3. TAILORED RETENTION OFFER**
One concrete offer customized to this customer's profile. Include specific numbers (discount %, free months, upgrade details).

**4. LONG-TERM RETENTION STRATEGY** (next 6 months)
Two actions to improve this customer's lifetime value over time.

**5. RISK SCORE INTERPRETATION**
One sentence explaining what the {prob:.1%} probability means in plain business language.

Be specific. Base everything on the actual customer profile. No generic advice."""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "ChurnGuard AI",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    model_ready = os.path.exists("outputs/model.pkl")
    return {
        "status": "ok" if model_ready else "model_not_trained",
        "model_loaded": model_ready,
        "analysis_plots": len(os.listdir(ANALYSIS_PATH)) if os.path.exists(ANALYSIS_PATH) else 0,
    }


@app.get("/get_analysis_images", tags=["Analysis"])
def get_images():
    if not os.path.exists(ANALYSIS_PATH):
        return {"images": []}
    images = sorted([
        f for f in os.listdir(ANALYSIS_PATH)
        if f.endswith((".jpg", ".png"))
    ])
    return {"images": images}


@app.get("/image/{image_name}", tags=["Analysis"])
def serve_image(image_name: str):
    path = os.path.join(ANALYSIS_PATH, image_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")
    return FileResponse(path)


@app.post("/predict", tags=["Prediction"])
def predict(data: RawCustomerData):
    """ML prediction only — fast, no AI call."""
    model = get_model()
    df = encode_customer(data.dict())
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[:, 1][0])

    return {
        "churn": prediction,
        "probability": round(probability, 4),
        "risk_level": get_risk_level(probability),
        "label": "WILL CHURN" if prediction == 1 else "WILL STAY",
    }


@app.post("/predict-with-recommendations", tags=["AI Recommendations"])
def predict_with_recommendations(data: RawCustomerData):
    """
    Full pipeline:
    1. ML model predicts churn probability
    2. Gemini AI generates a personalized retention strategy
    Returns both in one response.
    """
    # Step 1: ML prediction
    model = get_model()
    df = encode_customer(data.dict())
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[:, 1][0])

    pred_result = {
        "churn": prediction,
        "probability": round(probability, 4),
        "risk_level": get_risk_level(probability),
        "label": "WILL CHURN" if prediction == 1 else "WILL STAY",
    }

    # Step 2: Gemini AI retention strategy
    try:
        api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_AI_STUDIO_API_KEY not set in environment")

        # Initializing the modern Google GenAI Client
        client = genai.Client(api_key=api_key)
        prompt = build_retention_prompt(data.dict(), probability, prediction)

        # Using gemini-2.5-flash for balanced speed and strategic depth
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        ai_text = response.text

    except Exception as e:
        ai_text = f"⚠️ AI recommendations unavailable: {str(e)}"

    return {
        "prediction": pred_result,
        "ai_recommendations": ai_text,
    }