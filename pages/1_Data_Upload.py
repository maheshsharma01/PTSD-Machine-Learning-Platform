import streamlit as st
import pandas as pd
import os

# === Configuration: Set your label column name here ===
TARGET_COLUMN = 'ptsd_diagnosis'  # Change as per your dataset

st.set_page_config(page_title="Data Upload & Processing", layout="wide")
st.title("📂 Data Upload and Processing")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        st.write("**Dataset Shape:**", df.shape)

        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] > 1:
            st.subheader("Correlation Matrix (numeric features)")
            corr_matrix = numeric_df.corr()
            st.dataframe(
                corr_matrix.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1)
            )
        else:
            st.info("Not enough numeric columns for correlation matrix.")

        # Save processed data to file
        save_dir = os.path.join("data", "processed")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "processed_data.csv")
        df.to_csv(save_path, index=False)
        st.success(f"✅ Data processed and saved to '{save_path}'")

        # Save to session state for immediate use
        if TARGET_COLUMN in df.columns:
            st.session_state['features'] = df.drop(TARGET_COLUMN, axis=1)
            st.session_state['target'] = df[TARGET_COLUMN]
            st.session_state['data_loaded'] = True
            st.info("Processed data loaded into session for model training.")
        else:
            st.error(f"Target column '{TARGET_COLUMN}' not found in uploaded data.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
