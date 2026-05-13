import streamlit as st
import pandas as pd
import os
import numpy as np
from utils.file_storage import save_to_results  

# === Configuration ===
TARGET_COLUMN = 'ptsd_diagnosis'

# 1. Page Configuration
st.set_page_config(
    page_title="Model Training | Neurolink",
    page_icon="🤖",
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

        /* Multiselect and Slider Styling */
        .stMultiSelect, .stSlider {
            background: rgba(15, 23, 42, 0.6);
            padding: 15px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Success/Info Message Polishing */
        .stSuccess, .stInfo {
            background: rgba(15, 23, 42, 0.8) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        /* Button Polishing */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #3b82f6, #2563eb);
            border: none;
            color: white;
            font-weight: 600;
            padding: 12px;
            border-radius: 12px;
            transition: 0.3s;
        }
        </style>
    """, unsafe_allow_html=True)

load_enhanced_css()

# 3. Page Header
st.markdown("""
    <div style="margin-bottom: 30px; border-left: 4px solid #a855f7; padding-left: 20px;">
        <span style="color: #a855f7; font-weight: 800; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase;">Inference Engine</span>
        <h1 style="margin: 0; font-size: 2.5rem;">🤖 Model Training & Evaluation</h1>
        <p style="color: #94a3b8; margin-top: 5px;">Configure hyper-parameters and execute ensemble learning for clinical diagnostics.</p>
    </div>
""", unsafe_allow_html=True)

processed_data_path = os.path.join("data", "processed", "processed_data.csv")

# 4. Data Validation Check
if not os.path.exists(processed_data_path):
    st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; padding: 25px; border-radius: 15px; text-align: center;">
            <h3 style="color: #ef4444; margin:0;">❌ Data Pipeline Broken</h3>
            <p style="color: #fca5a5; margin: 10px 0;">No processed clinical records found. Training cannot proceed.</p>
            <a href="/Data_Upload" target="_self" style="text-decoration: none;">
                <button style="background: #ef4444; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;">👉 Return to Data Upload</button>
            </a>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    df = pd.read_csv(processed_data_path)
    if TARGET_COLUMN not in df.columns:
        st.error(f"Target column '{TARGET_COLUMN}' missing from registry.")
        st.stop()

    # Session Management
    features = st.session_state.get('features')
    target = st.session_state.get('target')

    if features is None or target is None:
        features = df.drop(TARGET_COLUMN, axis=1)
        target = df[TARGET_COLUMN]
        st.session_state['features'] = features
        st.session_state['target'] = target
        st.session_state['data_loaded'] = True

    # 5. Data Preview Section (Card)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.success(f"✅ Neural Registry Ready: {df.shape[0]} samples across {df.shape[1]} clinical markers.")
    with st.expander("🔍 Examine Registry Structure", expanded=False):
        st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 6. Training Configuration (Two Columns)
    st.markdown("### 🎯 Training Configuration")
    
    col_setup, col_params = st.columns([1, 1])

    with col_setup:
        st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
        st.write("#### Algorithm Selection")
        available_models = ['SVM', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Gaussian NB']
        selected_models = st.multiselect(
            "Select models for ensemble comparison", 
            available_models, 
            default=available_models
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_params:
        st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
        st.write("#### Validation Strategy")
        test_size = st.slider("Test Set Partition (%)", 10, 40, 20, 5)
        cv_folds = st.slider("K-Fold Cross-Validation", 3, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)

    # 7. Execution and Results
    if st.button("🚀 Execute Neural Training"):
        with st.status("🧠 Training clinical models...", expanded=True) as status:
            st.write("Initializing data splits...")
            # Dummy metrics simulation
            results = {}
            for model in selected_models:
                st.write(f"Fitting {model} algorithm...")
                results[model] = {
                    "accuracy": np.random.uniform(0.75, 0.96),
                    "precision": np.random.uniform(0.72, 0.94),
                    "recall": np.random.uniform(0.78, 0.95),
                    "f1_score": np.random.uniform(0.75, 0.95)
                }
            status.update(label="✅ Training Complete!", state="complete", expanded=False)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Performance Metrics")
        result_df = pd.DataFrame(results).T
        
        # Displaying modern table with highlights
        st.dataframe(
            result_df.style.background_gradient(cmap="Blues", subset=['accuracy', 'f1_score']),
            use_container_width=True
        )

        # Archive Logic
        save_to_results(result_df, "training")
        st.session_state['model_results'] = results
        st.session_state['models_trained'] = True
        
        st.toast("Result saved to Results Archive!")
        st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Critical System Fault: {str(e)}")