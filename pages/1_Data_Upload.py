import streamlit as st
import pandas as pd
import os

# === Configuration ===
TARGET_COLUMN = 'ptsd_diagnosis'  # Change as per your dataset

# 1. Page Configuration
st.set_page_config(
    page_title="Data Upload | Neurolink PTSD Platform", 
    page_icon="📂",
    layout="wide"
)

# 2. Enhanced UI Styling (Glassmorphism & Medical Aesthetic)
def load_enhanced_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Global Background & Font */
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
        h1, h2, h3 {
            background: linear-gradient(90deg, #60a5fa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }

        /* File Uploader Customization */
        section[data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.6);
            border: 2px dashed rgba(96, 165, 250, 0.3);
            border-radius: 15px;
            padding: 20px;
        }

        /* Metrics & Status Backgrounds */
        [data-testid="stMetric"], .stInfo {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
        }

        /* Improve Dataframe Appearance */
        [data-testid="stDataFrame"] {
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

load_enhanced_css()

# 3. Modern Header Section
st.markdown("""
    <div style="margin-bottom: 30px; border-left: 4px solid #60a5fa; padding-left: 20px;">
        <span style="color: #60a5fa; font-weight: 800; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase;">Neurolink Systems v2.5</span>
        <h1 style="margin: 0; font-size: 2.5rem;">📂 Data Upload & Processing</h1>
        <p style="color: #94a3b8; margin-top: 5px;">Secure clinical data ingestion for PTSD predictive modeling.</p>
    </div>
""", unsafe_allow_html=True)

# 4. File Upload Section
uploaded_file = st.file_uploader("Select Clinical CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Dataset Overview Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_prev, col_meta = st.columns([2, 1])
        
        with col_prev:
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
        with col_meta:
            st.subheader("🔍 Metadata Analysis")
            st.info(f"**Total Records:** {df.shape[0]}")
            st.info(f"**Total Features:** {df.shape[1]}")
            
            target_status = "✅ Found" if TARGET_COLUMN in df.columns else "❌ Missing"
            st.info(f"**Target ('{TARGET_COLUMN}'):** {target_status}")
            
            # Show data types breakdown
            st.write("**Feature Types:**")
            st.write(df.dtypes.value_counts())
        st.markdown('</div>', unsafe_allow_html=True)

        # Numeric Correlation Card
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] > 1:
            with st.expander("📈 View Feature Correlation Matrix", expanded=False):
                corr_matrix = numeric_df.corr()
                st.dataframe(
                    corr_matrix.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1),
                    use_container_width=True
                )
        else:
            st.warning("Not enough numeric columns for correlation analysis.")

        # Logic: Save processed data to file system
        save_dir = os.path.join("data", "processed")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "processed_data.csv")
        df.to_csv(save_path, index=False)

        # Logic: Update Session State for downstream tasks
        if TARGET_COLUMN in df.columns:
            st.session_state['features'] = df.drop(TARGET_COLUMN, axis=1)
            st.session_state['target'] = df[TARGET_COLUMN]
            st.session_state['data_loaded'] = True
            
            # Professional Success Box
            st.markdown(f"""
                <div style="background: rgba(52, 211, 153, 0.1); border: 1px solid #34d399; padding: 25px; border-radius: 15px; margin-top: 20px;">
                    <h3 style="color: #34d399; margin:0;">✅ Diagnostic Handshake Complete</h3>
                    <p style="color: #a7f3d0; margin: 10px 0 0 0; font-size: 1rem;">
                        The dataset has been successfully validated and loaded into the secure session. 
                        <b>Location:</b> <code>{save_path}</code>
                    </p>
                    <p style="color: #60a5fa; margin-top: 10px; font-weight: bold;">
                        👉 Proceed to 'Model Training' in the sidebar to begin analysis.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"Critical Error: Target column '{TARGET_COLUMN}' not found in uploaded CSV. Please check headers.")

    except Exception as e:
        st.error(f"❌ System Error during processing: {str(e)}")

else:
    # Empty State UI
    st.markdown("""
        <div style="text-align: center; padding: 80px 20px; background: rgba(30, 41, 59, 0.3); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1); margin-top: 20px;">
            <p style="color: #60a5fa; font-size: 1.2rem; font-weight: 600;">System Ready for Ingestion</p>
            <p style="color: #94a3b8;">Please drag and drop your clinical CSV file above to begin the predictive pipeline.</p>
        </div>
    """, unsafe_allow_html=True)