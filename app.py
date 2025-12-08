import os
import io
import pickle
from datetime import date, time

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Crime Case Closure Predictor",
    page_icon="🔒",
    layout="centered",
)

# -----------------------------
# Utilities
# -----------------------------
def to_age_group(v):
    """Map raw age to the numeric bins used in training."""
    try:
        v = int(v)
    except Exception:
        return 2
    if v < 18:
        return 0
    if v < 30:
        return 1
    if v < 50:
        return 2
    if v < 70:
        return 3
    return 4


@st.cache_resource
def load_model():
    """Load the trained SVM model from svm_model.pkl."""
    # Try pickle first, then joblib as a fallback
    try:
        with open("svm_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e1:
        try:
            import joblib  # lazy import; only if needed
            return joblib.load("svm_model.pkl")
        except Exception as e2:
            st.error("Failed to load 'svm_model.pkl'. If you upgraded scikit-learn, retrain and resave the model.")
            st.exception(e1)
            st.exception(e2)
            st.stop()


def read_expected_columns():
    """If feature_cols.txt exists (saved by your notebook), read expected feature order."""
    if os.path.exists("feature_cols.txt"):
        with open("feature_cols.txt", "r") as f:
            cols = [ln.strip() for ln in f if ln.strip()]
        return cols
    return None


def align_and_clean(df, expected_cols=None):
    """
    Align to expected feature order (if provided) and fill NaNs.
    Adds any missing expected columns as 0.
    """
    if expected_cols:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[expected_cols]
    # Replace inf and fill NaNs using column medians
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df


def predict_df(model, X):
    """Return predicted class and probability (if available)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return pred, proba
    # Fallback: no predict_proba
    pred = model.predict(X)
    return pred, None


# -----------------------------
# Load model and optional expected feature order
# -----------------------------
model = load_model()
EXPECTED_COLS = read_expected_columns()

# -----------------------------
# UI
# -----------------------------
st.title("🔒 Crime Case Closure Prediction")
st.caption("Single-model Streamlit app using your saved SVM model.")

tab_single, tab_batch, tab_about = st.tabs(["Single Prediction", "Batch Predictions", "About"])

# =============================
# Single prediction
# =============================
with tab_single:
    with st.form("case_form", clear_on_submit=False):
        st.subheader("Enter case details")

        c1, c2 = st.columns(2)
        with c1:
            date_reported = st.date_input("Date Reported", value=date(2020, 1, 1))
            time_reported = st.time_input("Time Reported", value=time(9, 0))
            victim_age = st.number_input("Victim Age", min_value=0, max_value=120, value=30, step=1)
            police_deployed = st.number_input("Police Deployed", min_value=0, max_value=1000, value=10, step=1)

        with c2:
            date_occ = st.date_input("Date of Occurrence", value=date(2020, 1, 1))
            time_occ = st.time_input("Time of Occurrence", value=time(12, 0))
            crime_desc = st.text_area("Crime Description (optional)", "", help="Used only to compute description word count.")

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Feature engineering to match training
        desc_word_count = len(crime_desc.split()) if crime_desc.strip() else 0

        features = {
            "report_hour": time_reported.hour,
            "report_dayofweek": date_reported.weekday(),
            "report_month": date_reported.month,
            "report_year": date_reported.year,
            "victim_age_group": to_age_group(victim_age),
            "desc_word_count": desc_word_count,
            "Police Deployed": police_deployed,
        }

        X = pd.DataFrame([features])
        X_aligned = align_and_clean(X.copy(), EXPECTED_COLS)

        try:
            pred, proba = predict_df(model, X_aligned)
        except Exception as e:
            st.error("Prediction failed. This usually means feature names/order do not match the model.")
            st.write("Features sent to model:")
            st.dataframe(X_aligned)
            if EXPECTED_COLS:
                st.write("Expected feature order from feature_cols.txt:")
                st.code("\n".join(EXPECTED_COLS))
            st.exception(e)
            st.stop()

        # Results
        st.divider()
        st.subheader("Results")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Prediction", "Closed" if int(pred[0]) == 1 else "Not Closed")
        with col_b:
            st.metric("Closure Probability", f"{float(proba[0]):.1%}" if proba is not None else "N/A")
        with col_c:
            risk = "Low Risk" if (proba is not None and float(proba[0]) >= 0.5) else "High Risk"
            st.metric("Risk Level", risk)

        with st.expander("View engineered features"):
            st.dataframe(X_aligned)

# =============================
# Batch predictions
# =============================
with tab_batch:
    st.subheader("Batch Predictions (CSV/XLSX)")
    st.caption(
        "Upload a file with the engineered feature columns. "
        "Use the template to ensure the correct names/order."
    )

    # Provide a downloadable template
    if EXPECTED_COLS:
        template_cols = EXPECTED_COLS
    else:
        template_cols = [
            "report_hour",
            "report_dayofweek",
            "report_month",
            "report_year",
            "victim_age_group",
            "desc_word_count",
            "Police Deployed",
        ]
    example = pd.DataFrame([{
        "report_hour": 12,
        "report_dayofweek": 3,
        "report_month": 6,
        "report_year": 2023,
        "victim_age_group": 2,
        "desc_word_count": 10,
        "Police Deployed": 5,
    }], columns=template_cols)

    csv_buf = io.StringIO()
    example.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download Template CSV",
        data=csv_buf.getvalue(),
        file_name="batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV or Excel with the engineered features", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
        except Exception as e:
            st.error("Failed to read the file. Ensure it's a valid CSV/XLSX.")
            st.exception(e)
            st.stop()

        st.write("Preview of uploaded data:")
        st.dataframe(df_in.head())

        # Align/clean
        Xb = align_and_clean(df_in.copy(), EXPECTED_COLS or template_cols)

        try:
            pred_b, proba_b = predict_df(model, Xb)
        except Exception as e:
            st.error("Batch prediction failed. Check column names/order against the template.")
            st.exception(e)
            st.stop()

        out = df_in.copy()
        out["prediction"] = pred_b.astype(int)
        if proba_b is not None:
            out["prob_closed"] = proba_b

        st.success("Batch predictions complete.")
        st.dataframe(out.head())

        # Download results
        out_csv = io.StringIO()
        out.to_csv(out_csv, index=False)
        st.download_button(
            label="Download Results CSV",
            data=out_csv.getvalue(),
            file_name="batch_results.csv",
            mime="text/csv",
        )

# =============================
# About / Help
# =============================
with tab_about:
    st.subheader("About this app")
    st.markdown(
        "- Uses a single SVM model saved as svm_model.pkl.\n"
        "- Expects engineered features (time-based, age group, description length, and police deployed).\n"
        "- If you changed scikit-learn versions recently and the model fails to load, "
        "open your notebook and retrain to regenerate svm_model.pkl in the current environment.\n"
    )
    st.markdown("Expected features (if feature_cols.txt is present, that order is enforced):")
    if EXPECTED_COLS:
        st.code("\n".join(EXPECTED_COLS))
    else:
        st.code(
            "\n".join(
                [
                    "report_hour",
                    "report_dayofweek",
                    "report_month",
                    "report_year",
                    "victim_age_group",
                    "desc_word_count",
                    "Police Deployed",
                ]
            )
        )