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

# Configure page first
st.set_page_config(
    page_title="PTSD ML Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Initialize core application components"""
    try:
        # Check database connection
        if hasattr(db_manager, 'is_connected') and db_manager.is_connected:
            st.session_state.db_connected = True
        else:
            st.session_state.db_connected = False
        
        # Initialize ML components (lazy loading)
        st.session_state.ml_models = None
        st.session_state.data_processor = None
        st.session_state.visualizer = None
        
    except Exception as e:
        st.session_state.db_connected = False
        st.warning(f"Component initialization warning: {str(e)}")

def render_header():
    """Render application header with status information"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title(f"🧠 {app_config.app_name}")
        st.markdown(f"**Version:** {app_config.app_version}")
    
    with col2:
        # Display connection status
        if st.session_state.get('db_connected', False):
            st.success("🔗 Connected")
        else:
            st.error("🔗 Disconnected")
    
    with col3:
        # Display session info
        if getattr(app_config, 'debug', False):
            with st.expander("🔧 Debug Info"):
                st.json({
                    "session_active": st.session_state.get('app_initialized', False),
                    "data_loaded": st.session_state.get('data_loaded', False),
                    "models_trained": st.session_state.get('models_trained', False)
                })

def render_sidebar():
    """Render enhanced sidebar with navigation and status"""
    st.sidebar.header("🚀 Navigation")
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

def render_main_dashboard():
    """Render the main dashboard content"""
    st.header("🚀 Welcome to the PTSD ML Platform")
    
    st.markdown("""
    ### Advanced Machine Learning Platform for PTSD Prediction
    
    This platform implements state-of-the-art machine learning algorithms for Post-Traumatic Stress Disorder prediction 
    based on comprehensive clinical research and validated assessment tools.
    """)
    
    # Feature highlights
    st.subheader("✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Data Processing**
        - PCL-5 Scale Analysis
        - Biomarker Integration
        - Data Quality Validation
        - Automated Feature Engineering
        """)
    
    with col2:
        st.markdown("""
        **🤖 ML Algorithms**
        - Support Vector Machines (SVM)
        - Artificial Neural Networks (ANN)
        - Random Forest & Gradient Boosting
        - Model Ensemble Methods
        """)
    
    with col3:
        st.markdown("""
        **📈 Analysis & Reporting**
        - Comprehensive Model Evaluation
        - Clinical Interpretation
        - Interactive Visualizations
        - Performance Monitoring
        """)
    
    # Quick start guide
    render_quick_start_guide()
    
    # System overview
    render_system_overview()

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
    """Render research background and methodology"""
    st.subheader("🔬 Research Background")
    
    st.markdown("""
    ### Clinical Foundation
    
    This platform is built on extensive research in PTSD prediction using machine learning approaches:
    
    - **PCL-5 Scale Integration**: Based on DSM-5 criteria for PTSD diagnosis
    - **Biomarker Analysis**: Incorporation of cortisol levels and other physiological markers
    - **Multi-Modal Approach**: Combining clinical, demographic, and biological data
    - **Validated Algorithms**: Implementations based on peer-reviewed research
    """)
    
    # Research metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **SVM Performance**
        - Accuracy: 82-90%
        - Best for neuroimaging data
        - Robust to overfitting
        """)
    
    with col2:
        st.info("""
        **ANN Results**
        - Accuracy: Up to 90%
        - Excellent for PCL-5 data
        - Complex pattern recognition
        """)
    
    with col3:
        st.info("""
        **Ensemble Methods**
        - Improved stability
        - Reduced prediction variance
        - Clinical reliability
        """)

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
    """Main application entry point"""
    try:
        # Initialize application
        initialize_application()
        
        # Render UI components
        render_header()
        render_sidebar()
        
        # Main content area
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🔬 Research", "📝 Sample Data"])
        
        with tab1:
            render_main_dashboard()
        
        with tab2:
            render_research_background()
        
        with tab3:
            render_sample_data_section()
        
        # Footer
        render_footer()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        
        # Fallback UI
        st.markdown("### ⚠️ Application Recovery Mode")
        st.markdown("The application encountered an error. Please refresh the page or contact support.")
        
        if st.button("🔄 Refresh Application"):
            st.rerun()

if __name__ == "__main__":
    main()