import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from utils.file_storage import list_results, delete_result

# 1. Page Configuration
st.set_page_config(
    page_title="Results Archive | Neurolink", 
    page_icon="📁", 
    layout="wide"
)

# 2. Enhanced UI Styling (Glassmorphism & Medical Aesthetic)
def load_enhanced_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Global Background */
        .stApp {
            background: radial-gradient(circle at top right, #0f172a, #020617);
            font-family: 'Inter', sans-serif;
        }
        
        /* High-Tech Card Style */
        .card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 25px;
            border-radius: 20px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }

        /* Gradient Headings */
        h1, h2, h3, h4 {
            background: linear-gradient(90deg, #60a5fa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }

        /* File Selectbox Polishing */
        .stSelectbox div[data-baseweb="select"] {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Dataframe Polishing */
        [data-testid="stDataFrame"] {
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Action Buttons */
        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            transition: 0.3s;
        }

        /* Delete Button Customization */
        div[data-testid="stVerticalBlock"] > div:last-child .stButton>button {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
        }
        
        div[data-testid="stVerticalBlock"] > div:last-child .stButton>button:hover {
            background: #ef4444;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

load_enhanced_css()

# 3. Modern Header Section
st.markdown("""
    <div style="margin-bottom: 30px; border-left: 4px solid #a855f7; padding-left: 20px;">
        <span style="color: #a855f7; font-weight: 800; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase;">Diagnostic Repository</span>
        <h1 style="margin: 0; font-size: 2.5rem;">📁 Results Archive</h1>
        <p style="color: #94a3b8; margin-top: 5px;">Secure management of historical clinical datasets and model inference logs.</p>
    </div>
""", unsafe_allow_html=True)

# 4. Logical Registry Check
files = list_results()

if not files:
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.3); padding: 50px; border-radius: 20px; text-align: center; border: 1px dashed rgba(255,255,255,0.1);">
            <p style="color: #94a3b8; font-size: 1.1rem;">No diagnostic results found in the secure registry.</p>
            <p style="color: #60a5fa; font-size: 0.9rem;">Run an inference session to generate new data.</p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# 5. File Selection Card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔍 Selection Interface")

# Format function to show human-readable dates
def format_file_info(f):
    mtime = os.path.getmtime(os.path.join('web_results', f))
    dt_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    return f"{f} (Modified: {dt_str})"

selected_file = st.selectbox(
    "Choose a diagnostic record to inspect:",
    files,
    format_func=format_file_info
)
st.markdown('</div>', unsafe_allow_html=True)

# 6. Preview and Content Management
file_path = os.path.join("web_results", selected_file)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"📄 Record Preview: {selected_file}")

try:
    if selected_file.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        st.dataframe(df, use_container_width=True)
    elif selected_file.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.json(data)
    else:
        st.warning("Encryption protocol mismatch: Unsupported file type.")
except Exception as e:
    st.error(f"Registry Access Violation: {e}")

# Action row for download and metadata
col_down, col_meta = st.columns([1, 1])

with col_down:
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    st.download_button(
        label="📥 Secure Data Export",
        data=file_bytes,
        file_name=selected_file,
        mime="text/csv" if selected_file.endswith(".csv") else "application/json",
        use_container_width=True
    )

with col_meta:
    st.info(f"**Size:** {os.path.getsize(file_path) / 1024:.2f} KB | **Type:** {selected_file.split('.')[-1].upper()}")

st.markdown('</div>', unsafe_allow_html=True)

# 7. Destruction Protocol
st.markdown("### ⚠️ Administrative Controls")
st.write("Permanent deletion of clinical records cannot be reversed.")

if st.button("🗑️ Delete Selected Record", type="primary", use_container_width=False):
    success = delete_result(selected_file)
    if success:
        st.success(f"Successfully deleted: {selected_file}")
        st.rerun()
    else:
        st.error(
            f"Unable to delete {selected_file}. System resource may be locked. "
            "Please ensure no other administrative instances are accessing this record."
        )