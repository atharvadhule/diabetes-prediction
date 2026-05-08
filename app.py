import streamlit as st
import pandas as pd
import joblib
import time

# --- Setup styling and page ---
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for Ambient Dark Medical UI & Single Card Layout
st.markdown("""
<style>
/* Base theme / ambient background */
.stApp {
    background: linear-gradient(135deg, #050b14 0%, #0a1128 50%, #0c1838 100%);
    color: #e0e1dd;
}

/* Hide top header bar natively added by Streamlit */
header {visibility: hidden;
        display : none;}

/* THE SINGLE MAIN BOX (Card) */
[data-testid="block-container"] {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(72, 202, 228, 0.1);
    padding: 0 4rem 4rem 4rem !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4), 0 0 40px rgba(72, 202, 228, 0.05);
    max-width: 700px !important;
    margin-bottom: 5vh;
    animation: fadeInCard 1.2s ease-out forwards;
    opacity: 0;
    transform: translateY(20px);
}

@keyframes fadeInCard {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInResult {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}

/* Headings */
h1 {
    text-align: center;
    color: #48cae4 !important;
    font-weight: 300 !important;
    letter-spacing: 1px;
    margin-bottom: 0rem !important;
    padding-bottom: 0.5rem !important;
}

/* UI Elements text color override */
p, label, .stMarkdown {
    color: #a9b3c4 !important;
    font-weight: 400;
}

/* Inputs styling with hover and focus effects */
.stTextInput>div>div>input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
    background-color: rgba(11, 19, 43, 0.4) !important;
    color: #e0e1dd !important;
    border: 1px solid rgba(72, 202, 228, 0.15) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
.stTextInput>div>div>input:focus, .stNumberInput input:focus, .stSelectbox [data-baseweb="select"]:focus {
    border-color: #48cae4 !important;
    box-shadow: 0 0 12px rgba(72, 202, 228, 0.25) !important;
    background-color: rgba(11, 19, 43, 0.7) !important;
}
.stTextInput>div>div>input:hover, .stNumberInput input:hover, .stSelectbox [data-baseweb="select"]:hover {
    border-color: rgba(72, 202, 228, 0.4) !important;
}

/* Styled Button */
div.stButton {
    display: flex;
    justify-content: center;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
div.stButton > button {
    background: linear-gradient(90deg, #0077b6 0%, #0096c7 100%);
    color: white !important;
    border-radius: 50px;
    border: none;
    padding: 0.6rem 3rem;
    font-size: 1.15rem;
    font-weight: 500;
    transition: all 0.4s ease;
    box-shadow: 0 4px 15px rgba(0, 119, 182, 0.3);
}
div.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(0, 150, 199, 0.6);
    transform: translateY(-2px);
    background: linear-gradient(90deg, #0096c7 0%, #48cae4 100%);
}

/* Results Section */
.result-card {
    animation: fadeInResult 0.6s ease-out forwards;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-top: 1rem;
}
.result-high {
    background: rgba(220, 53, 69, 0.08);
    border: 1px solid rgba(220, 53, 69, 0.25);
    color: #ff6b6b;
    box-shadow: 0 0 20px rgba(220, 53, 69, 0.15);
}
.result-low {
    background: rgba(40, 167, 69, 0.08);
    border: 1px solid rgba(40, 167, 69, 0.25);
    color: #51cf66;
    box-shadow: 0 0 20px rgba(40, 167, 69, 0.15);
}
</style>
""", unsafe_allow_html=True)

# 1. Main Heading (inside the block-container)
st.markdown("<h1>⚕️ Diabetes Risk Dashboard</h1>", unsafe_allow_html=True)


# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("diabetes_model.pkl")
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("Diagnostic Model 'diabetes_model.pkl' not found.")
    st.stop()


# 2. Input Form (Logical Medical Order)
col1, col2 = st.columns(2)

with col1:
    gender_str = st.selectbox("1. Gender", options=["Female", "Male", "Other"], index=0)
    bmi = st.number_input("3. BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    heart_disease_str = st.selectbox("5. Heart Disease", options=["No", "Yes"], index=0)
    blood_glucose_level = st.slider("7. Blood Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=5.0, step=0.1)

with col2:
    age = st.number_input("2. Age (Years)", min_value=1, max_value=120, value=35)
    hypertension_str = st.selectbox("4. Hypertension", options=["No", "Yes"], index=0)
    hba1c_level = st.slider("6. HbA1c Level (Hemoglobin A1c)", min_value=0.0, max_value=10.0, value=5.5, step=0.1)


# --- Process Inputs ---
hypertension = 1 if hypertension_str == "Yes" else 0
heart_disease = 1 if heart_disease_str == "Yes" else 0

if gender_str == "Female":
    gender_encoded = 0
elif gender_str == "Male":
    gender_encoded = 1
else:
    gender_encoded = 2

# Create prediction dataframe
input_df = pd.DataFrame([{
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "bmi": bmi,
    "blood_glucose_level": blood_glucose_level,
    "gender_encoded": gender_encoded,
    "HbA1c_level": hba1c_level
}])

# 3. Predict Button & 4. Results
if st.button("Predict Diabetes Risk"):
    with st.spinner("Analyzing parameters..."):
        time.sleep(1.0) # Slight delay for ambient feel
        try:
            prediction = model.predict(input_df)[0]
            
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0]
                confidence = prob[prediction] * 100
            else:
                conf_str = ""
            
            if prediction == 1:
                st.markdown(f'''
                    <div class="result-card result-high">
                        <h2 style="color: #ff6b6b; margin:0;">⚠️ High Risk Detected</h2>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="result-card result-low">
                        <h2 style="color: #51cf66; margin:0;">✅ Low Risk</h2>
                    </div>
                ''', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
