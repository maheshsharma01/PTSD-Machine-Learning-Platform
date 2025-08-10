import streamlit as st
import pandas as pd
import json
import os

from utils.file_storage import list_results, delete_result

st.set_page_config(page_title="📁 Results Archive", layout="wide")
st.title("📁 Results Archive")

st.markdown("""
This page lets you **view**, **download**, and **delete** saved results from the `web_results` folder.  
Results are saved automatically from other pages when using `save_to_results()`.
""")

files = list_results()

if not files:
    st.info("No results saved yet. Run a prediction or training session to generate results.")
    st.stop()

selected_file = st.selectbox(
    "Select a result file to view",
    files,
    format_func=lambda f: f"{f}  ({os.path.getmtime(os.path.join('web_results', f)):.0f})"
)

file_path = os.path.join("web_results", selected_file)

st.subheader(f"📄 Preview: {selected_file}")
try:
    if selected_file.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        st.dataframe(df, use_container_width=True)
    elif selected_file.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.json(data)
    else:
        st.warning("Unsupported file type.")
except Exception as e:
    st.error(f"Error reading file: {e}")

# Ensure file is closed before download
with open(file_path, "rb") as f:
    file_bytes = f.read()

st.download_button(
    label="📥 Download this file",
    data=file_bytes,
    file_name=selected_file,
    mime="text/csv" if selected_file.endswith(".csv") else "application/json"
)

st.markdown("---")

if st.button("🗑️ Delete this file", type="primary"):
    success = delete_result(selected_file)
    if success:
        st.success(f"Deleted: {selected_file}")
        st.experimental_rerun()
    else:
        st.error(
            f"Could not delete {selected_file}. "
            "Please ensure the file is not open in any other application. "
            "On Windows, try closing any programs using this file and restart the app."
        )
