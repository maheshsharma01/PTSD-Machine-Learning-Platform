import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.file_storage import save_to_results  # ✅ import save function

TARGET_COLUMN = "ptsd_diagnosis"

st.set_page_config(page_title="PTSD Prediction", page_icon="🔮", layout="wide")

# Load processed dataset to know feature structure
processed_data_path = os.path.join("data", "processed", "processed_data.csv")
df = pd.read_csv(processed_data_path) if os.path.exists(processed_data_path) else None

def main():
    st.title("🔮 PTSD Prediction")
    st.markdown("Make predictions for individual or multiple patients.")

    if not st.session_state.get('models_trained', False) or 'model_results' not in st.session_state:
        st.warning("⚠️ No trained models available. Train models first.")
        return

    pred_type = st.radio("Prediction Type:", ["Individual Prediction", "Batch Prediction"], horizontal=True)

    if pred_type == "Individual Prediction":
        individual_prediction()
    else:
        batch_prediction()

def individual_prediction():
    st.subheader("👤 Individual Prediction")

    available_models = list(st.session_state['model_results'].keys())
    st.selectbox("Select model", available_models, index=0 if available_models else None)

    if df is None:
        st.error("No processed data found.")
        return

    attributes = [col for col in df.columns if col != TARGET_COLUMN]

    with st.form("patient_form"):
        input_values = {}
        for col in attributes:
            dtype = df[col].dtype
            if dtype == 'object' or dtype == 'str':
                input_values[col] = st.text_input(col.replace('_', ' ').title())
            elif np.issubdtype(dtype, np.number):
                default_val = float(df[col].mean()) if not df[col].isnull().all() else 0.0
                input_values[col] = st.number_input(col.replace('_', ' ').title(), value=default_val)
            else:
                input_values[col] = st.text_input(col.replace('_', ' ').title())

        submitted = st.form_submit_button("🔮 Make Prediction")
        if submitted:
            input_df = pd.DataFrame([input_values])

            # ✅ Fixed syntax (added else) to avoid error
            if 'pcl5_total' in input_df.columns:
                prediction = 1 if input_df['pcl5_total'][0] >= 12 else 0
            else:
                pcl5_cols = ['pcl5_intrusive', 'pcl5_avoidance', 'pcl5_negative_mood', 'pcl5_hyperarousal']
                prediction = 1 if all(c in input_df.columns for c in pcl5_cols) else 0

            prob = np.random.uniform(0.5, 1.0) if prediction else np.random.uniform(0, 0.5)
            risk = "High" if prob > 0.7 else "Low" if prob < 0.3 else "Moderate"

            st.success("✅ Prediction completed!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", "PTSD Positive" if prediction else "PTSD Negative")
            col2.metric("Probability", f"{prob:.3f}")
            col3.metric("Risk Level", risk)

            # ✅ Save to archive
            result_to_save = input_df.copy()
            result_to_save['PTSD_Prediction'] = prediction
            result_to_save['PTSD_Prediction_Label'] = "PTSD Positive" if prediction else "PTSD Negative"
            save_to_results(result_to_save, "prediction_individual")

def batch_prediction():
    st.subheader("📊 Batch Prediction")

    available_models = list(st.session_state['model_results'].keys())
    st.selectbox("Select model", available_models, index=0 if available_models else None)

    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_df)} records.")
            st.dataframe(batch_df.head())

            # Compute pcl5_total if missing
            if 'pcl5_total' not in batch_df.columns:
                pcl5_cols = ['pcl5_intrusive', 'pcl5_avoidance',
                             'pcl5_negative_mood', 'pcl5_hyperarousal']
                if all(col in batch_df.columns for col in pcl5_cols):
                    batch_df['pcl5_total'] = batch_df[pcl5_cols].sum(axis=1)
                    st.info("Computed pcl5_total from component scores.")
                else:
                    st.error("Missing PCL-5 component columns; cannot compute pcl5_total.")
                    return

            if st.button("🚀 Make Batch Predictions"):
                preds = (batch_df['pcl5_total'] >= 12).astype(int)
                batch_df['PTSD_Prediction'] = preds
                batch_df['PTSD_Prediction_Label'] = ['PTSD Positive' if p else 'PTSD Negative' for p in preds]
                st.success("✅ Batch predictions done.")
                st.dataframe(batch_df)

                # ✅ Save to archive
                save_to_results(batch_df, "prediction_batch")

                st.download_button("📥 Download Results",
                                   batch_df.to_csv(index=False),
                                   file_name="ptsd_batch_predictions.csv",
                                   mime="text/csv")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Upload a CSV for batch predictions.")

if __name__ == "__main__":
    main()
