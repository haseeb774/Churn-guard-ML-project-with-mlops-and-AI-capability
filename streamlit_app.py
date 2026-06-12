"""
streamlit_app.py — ChurnGuard AI (Standalone)
Runs entirely on Streamlit Cloud — no FastAPI backend required.
ML model is loaded directly + Gemini AI called directly.

Deploy checklist:
  1. outputs/model.pkl  → committed to GitHub repo
  2. outputs/analysis/  → committed to GitHub repo (for EDA plots)
  3. Streamlit Cloud Secrets → GOOGLE_AI_STUDIO_API_KEY = "your-key"
  4. requirements.txt   → must include: streamlit, joblib, pandas, xgboost, google-genai
"""

import os
import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

.stApp { background: #0a0e17; }
h1,h2,h3 { font-family:'JetBrains Mono',monospace!important; color:#e6edf3; }
p,label,div { font-family:'DM Sans',sans-serif; }

.kpi-card {
    background: linear-gradient(135deg,#111827,#1a2235);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 22px 16px;
    text-align: center;
}
.kpi-val  { font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:700; line-height:1.1; }
.kpi-lbl  { font-size:.7rem; text-transform:uppercase; letter-spacing:2px; color:#6e7681; margin-top:6px; }

.ai-panel {
    background: #0d1117;
    border: 1px solid #1f6feb55;
    border-top: 3px solid #388bfd;
    border-radius: 12px;
    padding: 24px;
    margin-top: 16px;
}

.risk-tag {
    display:inline-block; padding:3px 12px;
    border-radius:20px; font-size:.72rem;
    font-weight:600; text-transform:uppercase; letter-spacing:1.2px;
}
.tag-HIGH   {background:rgba(248,81,73,.15); color:#f85149; border:1px solid #f8514955;}
.tag-MEDIUM {background:rgba(210,153,34,.15); color:#e3b341; border:1px solid #e3b34155;}
.tag-LOW    {background:rgba(63,185,80,.15);  color:#3fb950; border:1px solid #3fb95055;}

div[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }

.stButton>button {
    background:linear-gradient(135deg,#1a7f37,#2ea043);
    color:#fff; border:none; border-radius:8px;
    font-family:'JetBrains Mono',monospace;
    font-weight:700; font-size:.85rem;
    padding:12px 0; width:100%;
    transition:all .2s;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#2ea043,#3fb950);
    transform:translateY(-1px);
    box-shadow:0 4px 16px rgba(46,160,67,.35);
}

.section-label {
    font-size:.65rem; text-transform:uppercase;
    letter-spacing:2px; color:#388bfd;
    border-bottom:1px solid #21262d;
    padding-bottom:5px; margin-bottom:12px;
    font-family:'JetBrains Mono',monospace;
}

.status-ok  { color: #3fb950; font-weight: 600; }
.status-err { color: #f85149; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_PATH    = "outputs/model.pkl"
ANALYSIS_PATH = "outputs/analysis"

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


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


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


def ml_predict(customer: dict):
    model = load_model()
    if model is None:
        return None, None
    df = encode_customer(customer)
    prediction  = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[:, 1][0])
    return prediction, probability


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


def call_gemini(customer: dict, prob: float, prediction: int) -> str:
    api_key = st.secrets.get("GOOGLE_AI_STUDIO_API_KEY") or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        return "⚠️ GOOGLE_AI_STUDIO_API_KEY not set. Add it to Streamlit Cloud Secrets."

    try:
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-2.5-flash")
        prompt = build_retention_prompt(customer, prob, prediction)
        resp   = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛡️ ChurnGuard AI")
    st.caption("ML + AI customer retention platform")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔮 Predict & Advise", "📊 EDA Plots", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()

    # Status checks — no API call needed, just local file checks
    model_ok = os.path.exists(MODEL_PATH)
    api_key_set = bool(
        st.secrets.get("GOOGLE_AI_STUDIO_API_KEY")
        or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    )

    st.markdown(f"**ML Model** {'🟢 Ready' if model_ok else '🔴 Not found'}")
    st.markdown(f"**Gemini API** {'🟢 Key set' if api_key_set else '🔴 Key missing'}")

    if not model_ok:
        st.warning("`outputs/model.pkl` missing. Commit it to your repo.")
    if not api_key_set:
        st.warning("Add `GOOGLE_AI_STUDIO_API_KEY` in Streamlit Cloud → Settings → Secrets.")


# ── Page: Predict & Advise ────────────────────────────────────────────────────

if "🔮" in page:
    st.markdown("# 🔮 Predict & Advise")
    st.caption("Enter all customer fields, then run the analysis.")
    st.divider()

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
        gender      = st.selectbox("Gender", ["Female", "Male"])
        senior      = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner     = st.selectbox("Partner", ["Yes", "No"])
        dependents  = st.selectbox("Dependents", ["Yes", "No"])
        tenure      = st.number_input("Tenure (months)", 0, 120, 12)

        st.markdown('<div class="section-label" style="margin-top:16px">Account</div>', unsafe_allow_html=True)
        contract    = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless   = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment     = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly     = st.number_input("Monthly Charges ($)", 0.0, 500.0, 65.0, step=0.5)
        total       = st.number_input("Total Charges ($)", 0.0, 15000.0, 800.0, step=1.0)

    with c2:
        st.markdown('<div class="section-label">Phone Services</div>', unsafe_allow_html=True)
        phone       = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        st.markdown('<div class="section-label" style="margin-top:16px">Internet Services</div>', unsafe_allow_html=True)
        internet    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security    = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        backup      = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device      = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech        = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        tv          = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        movies      = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

    with c3:
        st.markdown('<div class="section-label">Analysis Options</div>', unsafe_allow_html=True)
        use_ai  = st.checkbox("🤖 Include AI Retention Strategy", value=True,
                              help="Calls Gemini AI to generate a personalized retention plan")
        st.markdown("")
        run_btn = st.button("⚡ ANALYZE CUSTOMER", use_container_width=True)

        st.divider()
        st.markdown("**High-Risk Signals**")
        st.markdown("""
        <small style="color:#6e7681">
        🔴 Month-to-month contract<br>
        🔴 Tenure &lt; 12 months<br>
        🟡 Electronic check payment<br>
        🟡 Fiber optic without security<br>
        🟡 No tech support<br>
        🟡 Monthly charges &gt; $70
        </small>
        """, unsafe_allow_html=True)

    # ── Run Analysis ──────────────────────────────────────────────────────────

    if run_btn:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model not found at `outputs/model.pkl`. Commit the trained model to your repo.")
            st.stop()

        customer = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi_lines,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": device,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly), "TotalCharges": float(total),
        }

        spinner_msg = "Running ML model + Gemini AI..." if use_ai else "Running ML model..."

        with st.spinner(spinner_msg):
            # Step 1: ML prediction
            prediction, probability = ml_predict(customer)

            if prediction is None:
                st.error("❌ Model failed to load.")
                st.stop()

            risk  = get_risk_level(probability)
            label = "WILL CHURN" if prediction == 1 else "WILL STAY"

            # Step 2: AI recommendations (optional)
            ai_text = None
            if use_ai:
                ai_text = call_gemini(customer, probability, prediction)

        st.divider()

        color = {"HIGH": "#f85149", "MEDIUM": "#e3b341", "LOW": "#3fb950"}[risk]

        # ── KPI Cards ─────────────────────────────────────────────────────────

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"""<div class="kpi-card">
            <div class="kpi-val" style="color:{color};">{label}</div>
            <div class="kpi-lbl">Prediction</div></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="kpi-card">
            <div class="kpi-val" style="color:{color};">{probability:.1%}</div>
            <div class="kpi-lbl">Churn Probability</div></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="kpi-card">
            <div class="kpi-val" style="color:{color};">{risk}</div>
            <div class="kpi-lbl">Risk Level</div></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="kpi-card">
            <div class="kpi-val" style="color:#8b949e;">{tenure}mo</div>
            <div class="kpi-lbl">Customer Tenure</div></div>""", unsafe_allow_html=True)

        # ── Risk Flags ────────────────────────────────────────────────────────

        flags = []
        if contract == "Month-to-month":             flags.append(("Month-to-month contract", "HIGH"))
        if tenure < 12:                              flags.append((f"Low tenure ({tenure}mo)", "HIGH"))
        if payment == "Electronic check":            flags.append(("Electronic check payment", "MEDIUM"))
        if internet == "Fiber optic" and security == "No":
                                                     flags.append(("Fiber optic + no security", "MEDIUM"))
        if tech == "No":                             flags.append(("No tech support", "MEDIUM"))
        if monthly > 70:                             flags.append((f"High charges (${monthly:.0f}/mo)", "MEDIUM"))
        if senior == 1:                              flags.append(("Senior citizen", "MEDIUM"))

        if flags:
            st.markdown("**Risk Factors Detected**")
            flag_html = " ".join(
                f'<span class="risk-tag tag-{lvl}">⚠ {lbl}</span>'
                for lbl, lvl in flags
            )
            st.markdown(flag_html, unsafe_allow_html=True)
        else:
            st.success("✅ No major risk flags detected for this customer.")

        # ── AI Recommendations ────────────────────────────────────────────────

        if ai_text:
            st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
            st.markdown("### 🤖 AI Retention Strategy")
            st.caption("Generated by Gemini AI — tailored to this customer's exact profile")
            st.markdown(ai_text)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Raw Result ────────────────────────────────────────────────────────

        with st.expander("🔧 Raw Prediction (JSON)"):
            st.json({
                "prediction": {
                    "churn": prediction,
                    "probability": round(probability, 4),
                    "risk_level": risk,
                    "label": label,
                },
                "ai_recommendations": ai_text or "Not requested",
            })


# ── Page: EDA Plots ───────────────────────────────────────────────────────────

elif "📊" in page:
    st.markdown("# 📊 EDA Analysis Plots")
    st.caption("Generated during model training. Commit `outputs/analysis/` to your repo.")
    st.divider()

    if not os.path.exists(ANALYSIS_PATH):
        st.info("No plots found. Run `python run_pipeline.py` locally, then commit `outputs/analysis/` to GitHub.")
    else:
        images = sorted([
            f for f in os.listdir(ANALYSIS_PATH)
            if f.endswith((".jpg", ".png"))
        ])

        if not images:
            st.info("No plot images found in `outputs/analysis/`.")
        else:
            for i in range(0, len(images), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(images):
                        name  = images[i + j]
                        path  = os.path.join(ANALYSIS_PATH, name)
                        clean = name.replace("_", " ").rsplit(".", 1)[0].title()
                        with col:
                            st.markdown(f"**{clean}**")
                            st.image(path, use_column_width=True)


# ── Page: About ───────────────────────────────────────────────────────────────

elif "ℹ️" in page:
    st.markdown("# ℹ️ About ChurnGuard AI")
    st.divider()

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("### What this is")
        st.markdown("""
        ChurnGuard AI is a **production-grade MLOps product** that:
        - Predicts customer churn probability (XGBoost + Optuna tuning)
        - Explains which signals drove the prediction
        - Generates a personalized retention strategy via Gemini AI
        - Logs all experiments to MLflow for reproducibility
        - Tracks data with DVC, containerized with Docker, CI/CD via GitHub Actions
        """)

        st.markdown("### Streamlit Cloud Setup")
        st.code("""
# 1. Commit your trained model
git add outputs/model.pkl -f
git add outputs/analysis/ -f
git commit -m "Add model and plots for deployment"
git push

# 2. In Streamlit Cloud → App Settings → Secrets:
GOOGLE_AI_STUDIO_API_KEY = "your-gemini-key-here"

# 3. requirements.txt must include:
# streamlit
# joblib
# pandas
# xgboost
# google-generativeai
        """, language="bash")

    with a2:
        st.markdown("### Tech Stack")
        stack = {
            "ML Model":             "XGBoost + Optuna hyperparameter tuning",
            "Experiment Tracking":  "MLflow",
            "AI Layer":             "Gemini 2.5 Flash (Google AI Studio)",
            "UI":                   "Streamlit (standalone, no FastAPI needed)",
            "Data Versioning":      "DVC",
            "Containerization":     "Docker",
            "CI/CD":                "GitHub Actions",
        }
        for k, v in stack.items():
            st.markdown(f"**{k}**: {v}")

        st.markdown("### Model Performance (approx)")
        st.markdown("""
        | Metric | Score |
        |--------|-------|
        | ROC-AUC | ~85% |
        | F1 Score (Churn) | ~60% |
        | Precision | ~67% |
        | Recall | ~54% |
        """)