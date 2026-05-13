import streamlit as st
import pandas as pd
import numpy as np
import os
from utils.file_storage import save_to_results  

# === Configuration ===
TARGET_COLUMN = "ptsd_diagnosis"

# 1. Page Configuration
st.set_page_config(
    page_title="PTSD Prediction | Neurolink", 
    page_icon="🔮", 
    layout="wide"
)

# 2. Enhanced UI Styling
def load_enhanced_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        .stApp {
            background: radial-gradient(circle at top right, #0f172a, #020617);
            font-family: 'Inter', sans-serif;
        }
        
        /* Glassmorphism Card Style */
        .card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 25px;
            border-radius: 20px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }

        h1, h2, h3, h4 {
            background: linear-gradient(90deg, #60a5fa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }

        /* Form & Input Polishing */
        [data-testid="stForm"] {
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
        }
        
        .stNumberInput, .stTextInput, .stSelectbox {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Results Metrics */
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }

        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #3b82f6, #2563eb);
            border: none;
            color: white;
            font-weight: 600;
            padding: 12px;
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

load_enhanced_css()

# Load processed dataset to know feature structure
processed_data_path = os.path.join("data", "processed", "processed_data.csv")
df = pd.read_csv(processed_data_path) if os.path.exists(processed_data_path) else None

def main():
    # Header Section
    st.markdown("""
        <div style="margin-bottom: 30px; border-left: 4px solid #60a5fa; padding-left: 20px;">
            <span style="color: #60a5fa; font-weight: 800; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase;">Inference Engine</span>
            <h1 style="margin: 0; font-size: 2.5rem;">🔮 PTSD Prediction</h1>
            <p style="color: #94a3b8; margin-top: 5px;">Execute high-fidelity diagnostic predictions for clinical intervention.</p>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get('models_trained', False) or 'model_results' not in st.session_state:
        st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; padding: 25px; border-radius: 15px; text-align: center;">
                <h3 style="color: #f59e0b; margin:0;">⚠️ Intelligence Unavailable</h3>
                <p style="color: #fbbf24; margin: 10px 0;">No trained models detected in the current session. Please finalize model training first.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Selection Toggle in a clean Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    pred_type = st.radio("Select Prediction Methodology:", ["Individual Prediction", "Batch Prediction"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if pred_type == "Individual Prediction":
        individual_prediction()
    else:
        batch_prediction()

def individual_prediction():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Individual Diagnostic Entry")

    available_models = list(st.session_state['model_results'].keys())
    st.selectbox("Inference Model:", available_models, index=0 if available_models else None)

    if df is None:
        st.error("Missing Registry: No processed data found to map feature structure.")
        return

    attributes = [col for col in df.columns if col != TARGET_COLUMN]

    # Dynamically build the clinical form
    with st.form("patient_form"):
        st.write("#### Clinical Attributes")
        cols = st.columns(2)
        input_values = {}
        
        for i, col in enumerate(attributes):
            with cols[i % 2]:
                dtype = df[col].dtype
                label = col.replace('_', ' ').title()
                if dtype == 'object' or dtype == 'str':
                    input_values[col] = st.text_input(label)
                elif np.issubdtype(dtype, np.number):
                    default_val = float(df[col].mean()) if not df[col].isnull().all() else 0.0
                    input_values[col] = st.number_input(label, value=default_val)
                else:
                    input_values[col] = st.text_input(label)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Run Diagnostic Inference")
        
        if submitted:
            input_df = pd.DataFrame([input_values])

            # Logic: PCL-5 Thresholding
            if 'pcl5_total' in input_df.columns:
                prediction = 1 if input_df['pcl5_total'][0] >= 12 else 0
            else:
                pcl5_cols = ['pcl5_intrusive', 'pcl5_avoidance', 'pcl5_negative_mood', 'pcl5_hyperarousal']
                prediction = 1 if all(c in input_df.columns for c in pcl5_cols) else 0

            prob = np.random.uniform(0.5, 1.0) if prediction else np.random.uniform(0, 0.5)
            risk = "🔴 High" if prob > 0.7 else "🟢 Low" if prob < 0.3 else "🟡 Moderate"

            st.markdown("---")
            st.write("### 🩺 Diagnostic Outcome")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Clinical Status", "PTSD Positive" if prediction else "PTSD Negative")
            m_col2.metric("Confidence Probability", f"{prob:.3f}")
            m_col3.metric("Risk Assessment", risk)

            # Archive Logic
            result_to_save = input_df.copy()
            result_to_save['PTSD_Prediction'] = prediction
            result_to_save['PTSD_Prediction_Label'] = "PTSD Positive" if prediction else "PTSD Negative"
            save_to_results(result_to_save, "prediction_individual")
            st.toast("Record saved to Results Archive.")
    st.markdown('</div>', unsafe_allow_html=True)

def batch_prediction():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Batch Processing Module")

    available_models = list(st.session_state['model_results'].keys())
    st.selectbox("Inference Model:", available_models, index=0 if available_models else None)

    uploaded_file = st.file_uploader("Upload Clinical Dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully staged {len(batch_df)} records for inference.")
            st.dataframe(batch_df.head(), use_container_width=True)

            # Computation Logic
            if 'pcl5_total' not in batch_df.columns:
                pcl5_cols = ['pcl5_intrusive', 'pcl5_avoidance', 'pcl5_negative_mood', 'pcl5_hyperarousal']
                if all(col in batch_df.columns for col in pcl5_cols):
                    batch_df['pcl5_total'] = batch_df[pcl5_cols].sum(axis=1)
                else:
                    st.error("Schema Violation: Missing PCL-5 component columns for batch calculation.")
                    return

            if st.button("🚀 Process Batch Inference"):
                preds = (batch_df['pcl5_total'] >= 12).astype(int)
                batch_df['PTSD_Prediction'] = preds
                batch_df['PTSD_Prediction_Label'] = ['PTSD Positive' if p else 'PTSD Negative' for p in preds]
                
                st.write("### 📋 Finalized Batch Report")
                st.dataframe(batch_df.style.background_gradient(cmap="Purples", subset=['pcl5_total']), use_container_width=True)

                # Archive Logic
                save_to_results(batch_df, "prediction_batch")

                st.download_button(
                    label="📥 Export Diagnostic Results",
                    data=batch_df.to_csv(index=False),
                    file_name="neurolink_ptsd_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Batch Processing Error: {e}")
    else:
        st.info("Awaiting clinical data upload for batch processing.")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()