import streamlit as st
import pandas as pd
import pickle

# =========================================================
# PAGE CONFIG (MUST BE FIRST)
# =========================================================
st.set_page_config(
    page_title="Cardio Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-section {
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# LOAD MODEL BUNDLE
# =========================================================
@st.cache_resource
def load_model_bundle():
    try:
        with open("models/final_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file not found. Please train and save the model first.")
        st.stop()

bundle = load_model_bundle()

model = bundle["model"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]
features = bundle["features"]

# =========================================================
# APP HEADER
# =========================================================
st.markdown("# 🫀 Cardiovascular Risk Prediction")
st.markdown("### ML-Powered Early Screening Tool")

with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    **Purpose:** This is a decision-support tool for early cardiovascular disease screening.
    
    **Model:** Recall-optimized Gradient Boosting classifier
    - Sensitivity: ~83% (catches most at-risk patients)
    - Decision Threshold: 0.36 (prioritizes minimizing false negatives)
    
    ⚠️ **Important:** This tool is NOT a medical diagnosis. Please consult healthcare professionals for clinical decisions.
    """)

st.divider()

# =========================================================
# SIDEBAR - Model Info
# =========================================================
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    **Model Type:** Calibrated Gradient Boosting
    
    **Performance:**
    - Recall: ~83%
    - Precision: ~66%
    - ROC-AUC: ~78%
    
    **Decision Threshold:** 0.36
    
    **Data:** ~65K patients
    """)
    
    st.divider()
    st.markdown("**⚠️ Disclaimer:** This is a prototype for demonstration purposes only.")

# =========================================================
# FEATURE LABELS & OPTIONS
# =========================================================
feature_labels = {
    "age": "Age (years)",
    "gender": "Gender",
    "height": "Height (cm)",
    "ap_hi": "Systolic Blood Pressure (mmHg)",
    "ap_lo": "Diastolic Blood Pressure (mmHg)",
    "cholesterol": "Cholesterol Level",
    "gluc": "Glucose Level",
    "smoke": "Smoking Status",
    "alco": "Alcohol Intake",
    "active": "Physical Activity",
    "bmi": "Body Mass Index"
}

categorical_options = {
    "gender": {"Male": 1, "Female": 0},
    "cholesterol": {
        "Normal (1)": 1,
        "Above Normal (2)": 2,
        "Well Above Normal (3)": 3
    },
    "gluc": {
        "Normal (1)": 1,
        "Above Normal (2)": 2,
        "Well Above Normal (3)": 3
    },
    "smoke": {"No": 0, "Yes": 1},
    "alco": {"No": 0, "Yes": 1},
    "active": {"No": 0, "Yes": 1}
}

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("## 📋 Patient Information")
st.markdown("*Please enter patient details below*")

user_input = {}
col1, col2 = st.columns(2)

# Height & weight handled separately (for BMI)
height_cm = 170.0
weight_kg = 70.0

for i, feature in enumerate(features):
    with col1 if i % 2 == 0 else col2:

        if feature == "bmi":
            continue

        if feature in categorical_options:
            selected = st.selectbox(
                feature_labels.get(feature, feature),
                list(categorical_options[feature].keys()),
                key=f"select_{feature}"
            )
            user_input[feature] = categorical_options[feature][selected]

        elif feature == "height":
            height_cm = st.number_input(
                "Height (cm)",
                min_value=140.0,
                max_value=210.0,
                value=170.0,
                step=0.5
            )
            user_input["height"] = height_cm

        else:
            user_input[feature] = st.number_input(
                feature_labels.get(feature, feature),
                min_value=0.0,
                step=1.0,
                key=f"input_{feature}"
            )

with col1:
    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=40.0,
        max_value=200.0,
        value=70.0,
        step=0.5
    )

# =========================================================
# BMI CALCULATION (AUTOMATIC)
# =========================================================
st.divider()

height_m = height_cm / 100
bmi = round(weight_kg / (height_m ** 2), 2)
user_input["bmi"] = bmi

col_bmi1, col_bmi2, col_bmi3 = st.columns([1, 2, 1])
with col_bmi2:
    st.info(
        f"📊 **Calculated BMI:** {bmi} kg/m² | "
        f"Height: {height_cm} cm | Weight: {weight_kg} kg"
    )

# Ensure correct feature order
input_df = pd.DataFrame([user_input])[features]

# =========================================================
# PREDICTION
# =========================================================
st.divider()

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button("🔍 Generate Risk Assessment", use_container_width=True)

if predict_button:
    try:
        X_scaled = scaler.transform(input_df)
        # Convert back to DataFrame to preserve feature names
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        prob = model.predict_proba(X_scaled_df)[:, 1][0]

        # Risk band for interpretation
        if prob < 0.20:
            risk_band = "Low"
            band_color = "green"
        elif prob < threshold:
            risk_band = "Moderate"
            band_color = "orange"
        else:
            risk_band = "High"
            band_color = "red"

        decision = "High Risk" if prob >= threshold else "Low Risk"

        st.markdown("## 📈 Risk Assessment Results")
        
        # Results in columns
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric("Risk Probability Score", f"{prob:.1%}")
        
        with result_col2:
            st.metric("Risk Classification", risk_band)

        st.divider()

        # Main decision box
        if decision == "High Risk":
            st.error(f"""
            ### ⚠️ HIGH CARDIOVASCULAR RISK
            
            **Risk Score:** {prob:.1%}  
            **Recommendation:** Further medical evaluation recommended
            
            This patient shows characteristics associated with elevated cardiovascular risk 
            and should undergo additional clinical assessment.
            """)
        else:
            st.success(f"""
            ###  NEGATIVE (Low Immediate Risk)
            
            **Risk Score:** {prob:.1%}  
            **Recommendation:** Continue routine health monitoring
            
            Current screening indicates low cardiovascular risk, but healthy lifestyle 
            practices remain important.
            """)

        # Additional details
        with st.expander("📊 Detailed Information"):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Model Details:**")
                st.write(f"- Decision Threshold: {threshold}")
                st.write(f"- Model Sensitivity: ~83%")
                st.write(f"- Model Specificity: ~65%")
            
            with detail_col2:
                st.markdown("**Risk Bands:**")
                st.write("- **Low Risk:** Score < 0.20")
                st.write(f"- **Moderate Risk:** 0.20 ≤ Score < {threshold}")
                st.write(f"- **High Risk:** Score ≥ {threshold}")

        # Medical disclaimer
        st.warning("""
        ⚠️ **Medical Disclaimer:**
        
        This tool is for **screening support only** and should NOT be used as a substitute 
        for professional medical diagnosis or treatment. Always consult with qualified 
        healthcare professionals for medical decisions.
        """)

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please ensure all input values are valid and within expected ranges.")
