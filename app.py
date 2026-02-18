"""
Streamlit app for Customer Churn Prediction with SHAP explanations.
"""

# Shim: make project root importable (helps when Streamlit changes cwd)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

from src.config import DATA_FILE, TARGET_COL
from src.deployment import predict_single, predict_batch, load_model
from src.preprocessing import data_load

# Cache loading sample data to build UI
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    df = data_load(DATA_FILE)
    return df


def build_input_form(df_sample: pd.DataFrame, prefix: str = "") -> Dict[str, Any]:
    """Build input widgets based on df_sample columns, with unique keys per prefix."""
    st.subheader("📝 Enter Customer Details")
    feature_cols = [c for c in df_sample.columns if c != TARGET_COL]

    input_data: Dict[str, Any] = {}
    for col in feature_cols:
        series = df_sample[col]

        if col == "SeniorCitizen":
            options = [0, 1]
            input_data[col] = st.selectbox(
                label=col,
                options=options,
                index=0,
                key=f"{prefix}{col}_select"
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            default = float(series.median()) if not series.isna().all() else 0.0
            input_data[col] = st.number_input(
                label=col,
                value=default,
                key=f"{prefix}{col}_num"
            )
        else:
            options = series.dropna().unique().tolist()
            if not options:
                options = ["Unknown"]
            input_data[col] = st.selectbox(
                label=col,
                options=options,
                index=0,
                key=f"{prefix}{col}_select"
            )
    return input_data


def explain_single(input_data: Dict[str, Any]) -> None:
    """Generate SHAP explanation for a single prediction."""
    model = load_model()
    df_input = pd.DataFrame([input_data])

    explainer = shap.Explainer(model.named_steps["clf"], model.named_steps["preprocess"].transform(df_input))
    shap_values = explainer(model.named_steps["preprocess"].transform(df_input))

    st.write("### 🔎 SHAP Explanation (Single Prediction)")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)


def explain_global(df_sample: pd.DataFrame) -> None:
    """Generate global SHAP summary plot for dataset."""
    model = load_model()
    X = df_sample.drop(columns=[TARGET_COL])

    explainer = shap.Explainer(model.named_steps["clf"], model.named_steps["preprocess"].transform(X))
    shap_values = explainer(model.named_steps["preprocess"].transform(X))

    st.write("### 🌍 SHAP Global Feature Importance")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features=model.named_steps["preprocess"].transform(X), show=False)
    st.pyplot(fig)


def main() -> None:
    st.title("📊 Customer Churn Prediction with SHAP")

    df_sample = load_sample_data()
    st.sidebar.markdown("### Data snapshot")
    st.sidebar.dataframe(df_sample.head(5))

    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "SHAP Explanation"])

    with tab1:
        input_data = build_input_form(df_sample, prefix="single_")
        st.write("Input preview:")
        st.json(input_data)

        if st.button("🔮 Predict Churn"):
            try:
                result = predict_single(input_data)
                churn_prob = result["churn_probability"]
                churn_class = "Churn" if result["prediction"] == 1 else "No Churn"

                st.write("**Predicted class:**", churn_class)
                st.write("**Churn probability:**", f"{churn_prob*100:.2f}%")

                if churn_prob > 0.5:
                    st.error("⚠️ High risk of churn")
                else:
                    st.success("✅ Low risk of churn")
            except Exception as exc:
                st.error("Prediction failed — check logs for details.")
                st.write(f"Error: {str(exc)}")

    with tab2:
        st.subheader("📂 Upload a cleaned CSV for batch predictions")
        uploaded = st.file_uploader("Choose a CSV", type=["csv"])
        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(uploaded_df.head())

            if st.button("🚀 Run batch prediction"):
                try:
                    result_df = predict_batch(uploaded_df)
                    st.write("Prediction results (first 10 rows):")
                    st.dataframe(result_df.head(10))

                    churn_rate = result_df["churn_prediction"].mean() * 100
                    st.write(f"**Batch churn rate:** {churn_rate:.2f}%")

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Download predictions",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as exc:
                    st.error("Batch prediction failed — check logs for details.")
                    st.write(f"Error: {str(exc)}")

    with tab3:
        st.subheader("🔎 Model Explanation with SHAP")
        input_data = build_input_form(df_sample, prefix="shap_")
        st.write("Input preview:")
        st.json(input_data)

        if st.button("Explain Single Prediction"):
            try:
                explain_single(input_data)
            except Exception as exc:
                st.error("SHAP explanation failed.")
                st.write(f"Error: {str(exc)}")

        if st.button("Show Global Feature Importance"):
            try:
                explain_global(df_sample)
            except Exception as exc:
                st.error("Global SHAP explanation failed.")
                st.write(f"Error: {str(exc)}")


if __name__ == "__main__":
    main()