
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Load models and transformers
@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return svm_model, rf_model, imputer, encoders

svm_model, rf_model, imputer, encoders = load_models()

# Streamlit UI
st.set_page_config(page_title="Crime Analytics Dashboard", layout="wide")
st.title("🚔 Indian Crime Case Prediction System")
st.markdown("**Predict case closure and optimal police deployment**")

# Sidebar inputs
st.sidebar.header("Input Case Details")

city = st.sidebar.selectbox("City", sorted(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']))
crime_domain = st.sidebar.selectbox("Crime Domain", sorted(['Theft', 'Assault', 'Fraud', 'Burglary', 'Robbery']))
victim_age = st.sidebar.slider("Victim Age", 0, 100, 35)
victim_gender = st.sidebar.selectbox("Victim Gender", ['Male', 'Female', 'Other'])
weapon_used = st.sidebar.selectbox("Weapon Used", ['None', 'Knife', 'Gun', 'Other'])
police_deployed = st.sidebar.slider("Police Deployed", 1, 20, 5)

# Preprocessing function (MUST match training pipeline)
def preprocess_input(city, crime_domain, victim_age, victim_gender, weapon_used, police_deployed):
    """Transform user input into model-ready features"""
    # Encode categorical variables
    city_encoded = encoders['city'].transform([city])[0]
    domain_encoded = encoders['domain'].transform([crime_domain])[0]
    gender_encoded = encoders['gender'].transform([victim_gender])[0]
    weapon_encoded = encoders['weapon'].transform([weapon_used])[0]
    
    # Create derived features (match training exactly)
    now = datetime.now()
    report_hour = now.hour
    report_dayofweek = now.weekday()
    report_month = now.month
    report_year = now.year
    occurrence_hour = report_hour  # Assume same day
    days_to_report = 0  # Assume immediate report
    victim_age_group = pd.cut([victim_age], bins=[0, 18, 30, 50, 70, 100], labels=[0, 1, 2, 3, 4])[0]
    desc_word_count = 10  # Default placeholder
    police_intensity = police_deployed / 20  # Normalize
    
    # Create DataFrame with EXACT same columns as training
    input_data = pd.DataFrame([{
        'city_encoded': city_encoded,
        'crime_code_encoded': 0,  # Default placeholder
        'weapon_encoded': weapon_encoded,
        'domain_encoded': domain_encoded,
        'gender_encoded': gender_encoded,
        'Victim Age': victim_age,
        'Police Deployed': police_deployed,
        'report_hour': report_hour,
        'report_dayofweek': report_dayofweek,
        'report_month': report_month,
        'report_year': report_year,
        'occurrence_hour': occurrence_hour,
        'days_to_report': days_to_report,
        'desc_word_count': desc_word_count,
        'victim_age_group': victim_age_group,
        'police_intensity': police_intensity
    }])
    
    return input_data

# Main tabs
tab1, tab2 = st.tabs(["🔮 Predictions", "📊 Model Performance"])

with tab1:
    st.subheader("Case Outcome Prediction")
    
    if st.button("Run Prediction", type="primary"):
        with st.spinner("Analyzing case..."):
            # Preprocess input
            input_df = preprocess_input(city, crime_domain, victim_age, victim_gender, weapon_used, police_deployed)
            
            # Apply imputer
            input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
            
            # Predict
            closure_prob = svm_model.predict_proba(input_imputed)[:, 1][0]
            optimal_police = rf_model.predict(input_imputed)[0]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Closure Probability", f"{closure_prob:.1%}")
            with col2:
                st.metric("Optimal Police", f"{optimal_police:.0f} officers")
            with col3:
                status = "High Risk" if closure_prob < 0.5 else "Low Risk"
                st.metric("Risk Level", status)
            
            # Risk indicator
            if closure_prob < 0.4:
                st.error("🔴 Low closure probability - Requires immediate attention")
            elif closure_prob < 0.7:
                st.warning("🟡 Moderate closure probability")
            else:
                st.success("🟢 High likelihood of closure")

with tab2:
    st.subheader("Model Performance Metrics")
    st.info("**SVM Classifier**: Predicts probability of case closure")
    st.info("**Random Forest**: Predicts optimal number of police officers")
    st.write("Models trained on historical crime data with time-based validation.")

# Footer
st.markdown("---")
st.markdown("*Deploy with: `streamlit run app.py`*")
```

---

### **How to Run:**

1. **Save the code above** into a file named `app.py` in the **same folder** as your `.pkl` files
2. **Open terminal** and navigate to that folder
3. **Run**: `streamlit run app.py`
4. **Interact** with your dashboard in the browser

**The app will load your trained models and provide real-time predictions.**