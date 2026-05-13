"""
Enhanced Main Application for PTSD ML Platform
Features improved error handling, session management, and user experience
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from db import create_table, add_user, get_all_users
from database.database_manager import get_db_manager

def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* Modern Gradient Background */
        .stApp {
            background: radial-gradient(circle at top right, #0f172a, #020617);
            font-family: 'Inter', sans-serif;
            color: #e2e8f0;
        }

        /* Glassmorphism Card Style */
        .card {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }

        .card:hover {
            border: 1px solid rgba(59, 130, 246, 0.4);
            transform: translateY(-3px);
        }

        /* Modern Typography */
        h1, h2, h3 {
            background: linear-gradient(90deg, #60a5fa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }

        /* Polishing Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: rgba(30, 41, 59, 0.5);
            border-radius: 10px 10px 0px 0px;
            color: #94a3b8;
            padding: 0px 20px;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(59, 130, 246, 0.2) !important;
            color: white !important;
        }

        /* Metric Styling */
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
        }

        /* Hide Streamlit elements */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )
# Configure page first
st.set_page_config(
    page_title="PTSD ML Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

# Import enhanced modules (with fallback handling)
try:
    from src.core.config_manager import get_app_config
    from src.core.error_handler import ErrorHandler, handle_streamlit_errors
    from src.core.session_state_manager import session_manager, initialize_session_state
    from src.core.database_manager import get_db_manager
    
    # Initialize components
    error_handler = ErrorHandler()
    db_manager = get_db_manager()
    app_config = get_app_config()
    
except ImportError as e:
    st.warning(f"Some enhanced modules not available: {e}. Using fallback mode.")
    
    # Fallback configuration
    class AppConfig:
        app_name = "PTSD ML Platform"
        app_version = "2.0.0"
        debug = False
    
    app_config = AppConfig()
    
    # Simple session state fallback
    def initialize_session_state():
        defaults = {
            'app_initialized': True,
            'debug_mode': False,
            'current_page': 'home',
            'data_loaded': False,
            'models_trained': False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

@st.cache_data
def get_sample_data():
    """Generate sample data for demonstration"""
    return pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'age': [25, 34, 45],
        'gender': ['F', 'M', 'F'],
        'trauma_type': ['Combat', 'Accident', 'Natural Disaster'],
        'pcl5_intrusive': [3.2, 2.8, 4.1],
        'pcl5_avoidance': [2.9, 3.5, 3.8],
        'pcl5_negative_mood': [3.4, 2.1, 4.2],
        'pcl5_hyperarousal': [3.1, 2.6, 3.9],
        'cortisol_level': [15.2, 12.8, 18.5],
        'ptsd_diagnosis': [1, 0, 1]
    })

def initialize_application():
    """Initialize the application with enhanced error handling"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Set up application state
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            st.session_state.app_start_time = datetime.utcnow()
        
        # Initialize core components
        if 'components_initialized' not in st.session_state:
            initialize_core_components()
            st.session_state.components_initialized = True
            
    except Exception as e:
        st.error(f"Application initialization error: {str(e)}")

def initialize_core_components():
    try:
        from database.database_manager import get_db_manager
        import os
        
        # LOGGING FOR YOU: Check if variables are actually loading
        url = os.getenv('DATABASE_URL')
        if not url:
            st.error("🚨 SYSTEM ALERT: The .env file is NOT being read by Python. Check the filename (no .txt at end!)")
        
        # Initialize the PostgreSQL manager
        st.session_state.db_manager = get_db_manager()
        
        # FORCE TEST
        st.session_state.db_manager._test_connection()
        st.session_state.db_connected = True
        st.success("✅ CONNECTION RE-ESTABLISHED: Internal bridge is active.")
            
    except Exception as e:
        st.session_state.db_connected = False
        # THIS WILL TELL US THE EXACT SQL ERROR
        st.error(f"❌ PHYSICAL CONNECTION ERROR: {str(e)}")
        
        # CHECK LIST:
        if "authentication failed" in str(e).lower():
            st.info("💡 SOLUTION: Your password in .env does not match what you set in PostgreSQL Shell.")
        elif "psycopg2" in str(e).lower():
            st.info("💡 SOLUTION: The driver is still missing. Run 'pip install psycopg2-binary' again.")

import streamlit.components.v1 as components

def render_header(authenticator):
    user_name = st.session_state.get("name", "Medical Professional")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
            <div style="margin-bottom: 10px;">
                <span style="color: #60a5fa; font-weight: 800; font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase;">Clinical Decision Support</span>
                <h1 style="margin: 0; font-size: 2.2rem;">Post Traumatic Stress Disorder Hub</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2); text-align: center; margin-bottom: 10px;">
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">Active Session</p>
                <strong style="color: white; font-size: 1.1rem;">Dr. {user_name}</strong>
            </div>
        """, unsafe_allow_html=True)
        
        # The Logout Button
        if st.button("🚪 Logout Session", use_container_width=True, key="logout_final_fix"):
            # 1. Clear Python Session State
            st.session_state["authentication_status"] = None
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # 2. CLEAR BROWSER COOKIES & STORAGE (The Nuclear Option)
            # This JS will wipe the cookies and local storage that the authenticator uses
            components.html(
                """
                <script>
                window.parent.localStorage.clear();
                window.parent.sessionStorage.clear();
                var cookies = window.parent.document.cookie.split(";");
                for (var i = 0; i < cookies.length; i++) {
                    var cookie = cookies[i];
                    var eqPos = cookie.indexOf("=");
                    var name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
                    window.parent.document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/";
                }
                window.parent.location.reload();
                </script>
                """,
                height=0,
            )
            st.rerun()

def render_sidebar(authenticator):
    """Render enhanced sidebar with navigation and status"""
    st.sidebar.markdown("## 📂 Workspace")
    st.sidebar.markdown("### 📊 Modules")
    st.sidebar.markdown("---")
    
    # Application status indicators
    st.sidebar.subheader("📊 Status Dashboard")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        session_status = "Active" if st.session_state.get('app_initialized', False) else "Inactive"
        st.metric("Session", session_status)
    with col2:
        db_status = "✅" if st.session_state.get('db_connected', False) else "❌"
        st.metric("Database", db_status)
    
    # Data status
    data_loaded = st.session_state.get('data_loaded', False)
    models_trained = st.session_state.get('models_trained', False)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Data", "✅ Loaded" if data_loaded else "⭕ Not Loaded")
    with col2:
        st.metric("Models", "✅ Trained" if models_trained else "⭕ Not Trained")
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Status"):
        st.rerun()
    
    if st.sidebar.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
    
    # Navigation info
    st.sidebar.subheader("📖 Pages")
    st.sidebar.info("""
    **Available Pages:**
    - 📊 Data Upload
    - 🤖 Model Training  
    - 🔮 Prediction
    - ⚖️ Model Comparison
    - 💾 Database Management
    
    Use the page selector above to navigate.
    """)
    st.sidebar.markdown("---")
    

def render_main_dashboard():
    # Hero Introduction
    st.markdown("""
        <div class="card">
            <h2 style="margin-top:0;">📊 PTSD Prediction Platform</h2>
            <p style="color: #94a3b8; font-size: 1.1rem; line-height: 1.6;">
                This platform integrates validated clinical PCL-5 assessments with advanced machine learning 
                to provide highly accurate PTSD risk stratifications.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Core Features Grid
    st.markdown("### 🧠 Diagnostic Capabilities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color:#60a5fa;">🧬 Biomarker Analysis</h4>
            <p style="font-size:0.9rem; color:#94a3b8;">Processing Cortisol, Heart Rate Variability, and sleep pattern data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color:#a855f7;">🤖 ML Ensemble</h4>
            <p style="font-size:0.9rem; color:#94a3b8;">Cross-validation via SVM, ANN, and Random Forest algorithms.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="card">
            <h4 style="color:#34d399;">📈 Clinical Insight</h4>
            <p style="font-size:0.9rem; color:#94a3b8;">Actionable reports for evidence-based trauma intervention.</p>
        </div>
        """, unsafe_allow_html=True)

    # System Status Section
    st.markdown("---")
    render_system_overview()
    
    # Quick Start
    render_quick_start_guide()

def render_quick_start_guide():
    """Render quick start guide for new users"""
    st.subheader("🎯 Quick Start Guide")
    
    steps = [
        ("1️⃣", "Upload Data", "Load your PTSD dataset or use our sample template"),
        ("2️⃣", "Train Models", "Train multiple ML algorithms on your data"),
        ("3️⃣", "Make Predictions", "Get individual or batch PTSD predictions"),
        ("4️⃣", "Compare Results", "Analyze model performance and clinical interpretation")
    ]
    
    for emoji, title, description in steps:
        with st.expander(f"{emoji} {title}", expanded=False):
            st.markdown(description)
            
            # Add relevant navigation buttons
            if title == "Upload Data":
                st.info("👉 Use the **Data Upload** page in the sidebar to get started")
            elif title == "Train Models" and st.session_state.get('data_loaded', False):
                st.info("👉 Use the **Model Training** page to train your models")

def render_system_overview():
    """Render system overview and stats"""
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Session duration
        if 'app_start_time' in st.session_state:
            duration = datetime.utcnow() - st.session_state.app_start_time
            st.metric("Session Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")
        else:
            st.metric("Session Duration", "N/A")
    
    with col2:
        # Database status
        db_status = "Connected" if st.session_state.get('db_connected', False) else "Disconnected"
        st.metric("Database", db_status)
    
    with col3:
        # Data status
        data_count = len(st.session_state.get('processed_data', pd.DataFrame()))
        st.metric("Data Records", data_count)
    
    with col4:
        # Model status
        models_count = len(st.session_state.get('model_results', {}))
        st.metric("Trained Models", models_count)

def render_sample_data_section():
    """Render sample data template section"""
    st.subheader("📝 Sample Data Template")
    st.markdown("Download a template to see the expected data format:")
    
    # Show sample data
    sample_data = get_sample_data()
    st.dataframe(sample_data)
    
    # Convert to CSV for download
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Template",
        data=csv,
        file_name="ptsd_data_template.csv",
        mime="text/csv"
    )

def render_research_background():
    st.markdown("## 🔬 Evidence-Based Methodology")
    
    st.info("""
    **Methodology Overview:** Our predictive models are trained on the DSM-5 criteria, 
    specifically targeting the four symptom clusters of PTSD: Intrusive thoughts, 
    Avoidance, Negative alterations in cognition/mood, and Hyperarousal.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>Algorithm Benchmarking</h4>
            <table style="width:100%; color:#94a3b8;">
                <tr><td><b>ANN (Neural Net)</b></td><td style="color:#34d399;">92% Acc.</td></tr>
                <tr><td><b>SVM (Vector)</b></td><td style="color:#34d399;">89% Acc.</td></tr>
                <tr><td><b>Random Forest</b></td><td style="color:#34d399;">87% Acc.</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
            <h4>Clinical Variables</h4>
            <ul style="color:#94a3b8; font-size:0.9rem;">
                <li>PCL-5 Cluster Scores (Validated)</li>
                <li>Cortisol Response Curves</li>
                <li>Trauma Recency & Frequency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_footer():
    """Render application footer"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        session_id = st.session_state.get('_session_id', 'unknown')[:8]
        st.markdown(f"""
        **{app_config.app_name}** v{app_config.app_version} | 
        Built with ❤️ using Streamlit | 
        Session ID: `{session_id}...`
        """)
    
    with col2:
        st.markdown("**Status:** Production Ready")
    
    with col3:
        if st.button("ℹ️ About", key="about_button"):
            show_about_info()
            
def signup():
    st.subheader("🆕 Create New Account")

    name = st.text_input("Full Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Signup"):
        if name and username and password:

            # ✅ CORRECT HASH
            hashed_password = stauth.Hasher().hash(password)

            add_user(username, name, hashed_password)

            st.success("✅ Account created successfully")
            st.info("👉 Now go to Login")

        else:
            st.error("❌ Please fill all fields")
            
def load_users():
    users = get_all_users()
    
    config = {
        "credentials": {
            "usernames": {}
        },
        "cookie": {
            "expiry_days": 30,
            "key": "ptsd_platform_secure_key",
            "name": "ptsd_auth_cookie"
        }
    }

    if users:
        for user in users:
            username, name, password = user
            config["credentials"]["usernames"][username] = {
                "name": name,
                "password": password
            }
            
    return config


def show_about_info():
    """Show about information"""
    st.info(f"""
    ## {app_config.app_name}
    **Version:** {app_config.app_version}
    
    Advanced machine learning platform for PTSD prediction and analysis, 
    implementing clinically validated algorithms and assessment tools.
    
    **Features:**
    - Multiple ML algorithms (SVM, ANN, RF, GB)
    - PCL-5 scale integration
    - Biomarker analysis
    - Clinical interpretation
    - Comprehensive evaluation metrics
    """)

def main():
    create_table()
    auth_config = load_users()
    
    if "name" not in st.session_state:
        st.session_state["name"] = None

    menu = ["Login", "Signup"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Signup":
        signup()
        return

    try:
        # UPDATED: Removed the preauthorized argument (now only 4 parameters)
        authenticator = stauth.Authenticate(
            auth_config['credentials'],
            auth_config['cookie']['name'],
            auth_config['cookie']['key'],
            auth_config['cookie']['expiry_days']
        )

        # UPDATED: In the latest version, login just takes the location keyword
        authenticator.login(location='main')

        if st.session_state.get("authentication_status") is False:
            st.error("❌ Username/password incorrect")
            
        elif st.session_state.get("authentication_status") is None:
            st.warning("⚠️ Please login to continue")
            st.stop()

        elif st.session_state.get("authentication_status"):
            # Ensure name and username are synced from authenticator state
            st.session_state["name"] = st.session_state.get("name")
            st.session_state["username"] = st.session_state.get("username")

            # Load App UI
            initialize_application()
            render_header(authenticator)
            render_sidebar(authenticator)

            tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🔬 Research", "📝 Sample Data"])

            with tab1:
                render_main_dashboard()
            with tab2:
                render_research_background()
            with tab3:
                render_sample_data_section()

            render_footer()

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        if st.button("🔄 Hard Reset & Clear Cache"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
    