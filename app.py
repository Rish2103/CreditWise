import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Page Config
st.set_page_config(page_title="CreditWise: Loan Approval Predictor", layout="wide")

# 1. LOAD AND PREPROCESS DATA (Cached to run once)
@st.cache_resource
def train_model():
    df = pd.read_csv("loan_approval_data.csv")
    
    # Identify columns
    categorical_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    numerical_cols = ["Applicant_Income", "Coapplicant_Income", "Age", "Dependents", "Credit_Score", "Existing_Loans", "DTI_Ratio", "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term"]
    
    # Impute missing values
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
    
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols + ["Education_Level"]] = cat_imp.fit_transform(df[categorical_cols + ["Education_Level"]])
    
    # Encoding
    le_edu = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    
    le_target = LabelEncoder()
    df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])
    
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    ohe_data = ohe.fit_transform(df[categorical_cols])
    ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(categorical_cols))
    
    # Combine features
    X = pd.concat([df[numerical_cols + ["Education_Level"]], ohe_df], axis=1)
    y = df["Loan_Approved"]
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Model
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, ohe, le_edu, categorical_cols, numerical_cols

model, scaler, ohe, le_edu, cat_cols, num_cols = train_model()

# 2. STREAMLIT FRONTEND
st.title("🏦 CreditWise Loan Approval Predictor")
st.markdown("Enter the applicant's details below to predict loan eligibility.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        income = st.number_input("Applicant Income ($)", value=5000)
        co_income = st.number_input("Co-applicant Income ($)", value=0)
        age = st.slider("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
    with col2:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        loan_amt = st.number_input("Loan Amount ($)", value=15000)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84])
        marital = st.selectbox("Marital Status", ["Single", "Married"])
        
    with col3:
        dependents = st.number_input("Dependents", 0, 10, 0)
        savings = st.number_input("Total Savings ($)", value=1000)
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

    # Additional inputs
    col4, col5 = st.columns(2)
    with col4:
        emp_status = st.selectbox("Employment Status", ['Salaried', 'Self-employed', 'Contract', 'Unemployed'])
        loan_purpose = st.selectbox("Loan Purpose", ['Personal', 'Car', 'Business', 'Home', 'Education'])
    with col5:
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
        collateral = st.number_input("Collateral Value ($)", value=0)

    submit = st.form_submit_button("Predict Approval Status")

# 3. PREDICTION LOGIC
if submit:
    # Prepare input data
    input_data = pd.DataFrame({
        "Applicant_Income": [income], "Coapplicant_Income": [co_income], "Age": [age],
        "Dependents": [dependents], "Credit_Score": [credit_score], "Existing_Loans": [0], 
        "DTI_Ratio": [dti], "Savings": [savings], "Collateral_Value": [collateral],
        "Loan_Amount": [loan_amt], "Loan_Term": [loan_term],
        "Education_Level": [le_edu.transform([education])[0]]
    })
    
    # OHE for categorical
    cat_input = pd.DataFrame({
        "Employment_Status": [emp_status], "Marital_Status": [marital],
        "Loan_Purpose": [loan_purpose], "Property_Area": [property_area],
        "Gender": [gender], "Employer_Category": ["Private"] # Defaulting based on common values
    })
    
    ohe_input = ohe.transform(cat_input)
    ohe_input_df = pd.DataFrame(ohe_input, columns=ohe.get_feature_names_out(cat_cols))
    
    # Combine and scale
    final_input = pd.concat([input_data, ohe_input_df], axis=1)
    final_input_scaled = scaler.transform(final_input)
    
    # Predict
    prediction = model.predict(final_input_scaled)
    prob = model.predict_proba(final_input_scaled)[0][1]
    
    if prediction[0] == 1:
        st.success(f"✅ Approved! (Confidence: {prob:.2%})")
    else:
        st.error(f"❌ Denied (Confidence: {1-prob:.2%})")
