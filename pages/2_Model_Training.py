import streamlit as st
import pandas as pd
import os
import numpy as np
from utils.file_storage import save_to_results  # ✅ import function

TARGET_COLUMN = 'ptsd_diagnosis'  # change if needed

st.set_page_config(page_title="Model Training & Evaluation", layout="wide")
st.title("🤖 Model Training & Evaluation")

processed_data_path = os.path.join("data", "processed", "processed_data.csv")

if not os.path.exists(processed_data_path):
    st.error("❌ No processed data found. Please upload and process data first.")
    st.info("👉 Go to the Data Upload page.")
    st.stop()

try:
    df = pd.read_csv(processed_data_path)
    if TARGET_COLUMN not in df.columns:
        st.error(f"Target column '{TARGET_COLUMN}' not found in data.")
        st.stop()

    features = st.session_state.get('features')
    target = st.session_state.get('target')

    if features is None or target is None:
        features = df.drop(TARGET_COLUMN, axis=1)
        target = df[TARGET_COLUMN]
        st.session_state['features'] = features
        st.session_state['target'] = target
        st.session_state['data_loaded'] = True

    st.success("✅ Data loaded for training.")
    st.write(f"**Data Shape:** {df.shape}")
    st.dataframe(df.head())

    st.header("🎯 Model Training Setup")
    available_models = ['SVM', 'Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Gaussian NB']
    selected_models = st.multiselect(
        "Select models to train", available_models, default=available_models)

    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

    if st.button("🏃 Start Training"):
        st.info("Training started...")
        # Dummy metrics simulation — replace with real training later
        results = {}
        for model in selected_models:
            results[model] = {
                "accuracy": np.random.uniform(0.7, 0.95),
                "precision": np.random.uniform(0.7, 0.95),
                "recall": np.random.uniform(0.7, 0.95),
                "f1_score": np.random.uniform(0.7, 0.95)
            }

        st.success("✅ Training complete!")
        result_df = pd.DataFrame(results).T
        st.subheader("📊 Training Results")
        st.dataframe(result_df.style.highlight_max(axis=0))

        # ✅ Save results to archive
        save_to_results(result_df, "training")

        # ✅ Store in session state for other pages
        st.session_state['model_results'] = results
        st.session_state['models_trained'] = True

except Exception as e:
    st.error(f"Error loading data: {e}")
