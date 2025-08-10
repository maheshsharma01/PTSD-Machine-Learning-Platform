import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from database.database_manager import DatabaseManager
import json
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Database Management - PTSD ML Platform",
    page_icon="💾",
    layout="wide"
)

def main():
    st.title("💾 Database Management")
    st.markdown("Manage patient data, model results, and prediction history stored in the database.")
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Sidebar navigation
    st.sidebar.header("Database Operations")
    operation = st.sidebar.selectbox(
        "Select Operation:",
        ["Overview", "Patient Data", "Model Results", "Prediction History", "Analytics", "Data Export"]
    )
    
    if operation == "Overview":
        show_overview()
    elif operation == "Patient Data":
        manage_patient_data()
    elif operation == "Model Results":
        manage_model_results()
    elif operation == "Prediction History":
        manage_prediction_history()
    elif operation == "Analytics":
        show_analytics()
    elif operation == "Data Export":
        data_export()

def show_overview():
    """Display database overview and statistics."""
    
    st.header("📊 Database Overview")
    
    try:
        # Get database statistics
        stats = st.session_state.db_manager.get_database_stats()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", stats.get('total_patients', 0))
        
        with col2:
            st.metric("Total Predictions", stats.get('total_predictions', 0))
        
        with col3:
            st.metric("Trained Models", stats.get('total_models', 0))
        
        with col4:
            ptsd_positive = stats.get('ptsd_positive_cases', 0)
            ptsd_negative = stats.get('ptsd_negative_cases', 0)
            total_diagnosed = ptsd_positive + ptsd_negative
            if total_diagnosed > 0:
                ptsd_rate = (ptsd_positive / total_diagnosed) * 100
                st.metric("PTSD Rate", f"{ptsd_rate:.1f}%")
            else:
                st.metric("PTSD Rate", "N/A")
        
        # Database status
        st.subheader("🔌 Database Status")
        if stats:
            st.success("✅ Database connection active")
            
            # Show table information
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Patient Records:** {stats.get('total_patients', 0)}")
                st.info(f"**PTSD Positive:** {stats.get('ptsd_positive_cases', 0)}")
                st.info(f"**PTSD Negative:** {stats.get('ptsd_negative_cases', 0)}")
            
            with col2:
                st.info(f"**Model Results:** {stats.get('total_models', 0)}")
                st.info(f"**Predictions Made:** {stats.get('total_predictions', 0)}")
        else:
            st.error("❌ Database connection failed")
        
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")

def manage_patient_data():
    """Manage patient data in the database."""
    
    st.header("👥 Patient Data Management")
    
    # Search and filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_patient_id = st.text_input("Search by Patient ID:", placeholder="e.g., P001")
    
    with col2:
        limit = st.number_input("Records to display:", min_value=10, max_value=1000, value=100)
    
    with col3:
        show_ptsd_only = st.checkbox("Show PTSD positive only")
    
    # Retrieve patient data
    try:
        if search_patient_id:
            df = st.session_state.db_manager.get_patient_data(patient_id=search_patient_id)
        else:
            df = st.session_state.db_manager.get_patient_data(limit=limit)
        
        if not df.empty:
            # Filter for PTSD positive if requested
            if show_ptsd_only:
                df = df[df['ptsd_diagnosis'] == 1]
            
            st.success(f"Found {len(df)} patient records")
            
            # Display summary statistics
            if len(df) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_age = df['age'].mean() if 'age' in df.columns else 0
                    st.metric("Average Age", f"{avg_age:.1f}" if avg_age > 0 else "N/A")
                
                with col2:
                    if 'gender' in df.columns:
                        female_count = (df['gender'] == 'F').sum()
                        male_count = (df['gender'] == 'M').sum()
                        st.metric("Female/Male", f"{female_count}/{male_count}")
                    else:
                        st.metric("Female/Male", "N/A")
                
                with col3:
                    if 'pcl5_total_score' in df.columns:
                        avg_pcl5 = df['pcl5_total_score'].mean()
                        st.metric("Avg PCL-5 Score", f"{avg_pcl5:.1f}" if not pd.isna(avg_pcl5) else "N/A")
                    else:
                        st.metric("Avg PCL-5 Score", "N/A")
                
                with col4:
                    if 'ptsd_diagnosis' in df.columns:
                        ptsd_rate = (df['ptsd_diagnosis'] == 1).mean() * 100
                        st.metric("PTSD Rate", f"{ptsd_rate:.1f}%")
                    else:
                        st.metric("PTSD Rate", "N/A")
            
            # Display data table
            st.subheader("📋 Patient Records")
            
            # Format datetime columns
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            if 'updated_at' in df.columns:
                df['updated_at'] = pd.to_datetime(df['updated_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Patient Data",
                data=csv,
                file_name=f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("No patient records found.")
    
    except Exception as e:
        st.error(f"Error retrieving patient data: {str(e)}")

def manage_model_results():
    """Manage model training results."""
    
    st.header("🤖 Model Results Management")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        model_filter = st.selectbox(
            "Filter by model:",
            ["All Models", "SVM", "ANN", "Decision_Tree", "Gaussian_NB", "Random_Forest", "Gradient_Boosting"]
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Training Date", "Accuracy", "F1 Score", "AUC Score"]
        )
    
    try:
        # Get model results
        model_name = None if model_filter == "All Models" else model_filter
        df = st.session_state.db_manager.get_model_results(model_name=model_name)
        
        if not df.empty:
            st.success(f"Found {len(df)} model training records")
            
            # Sort data
            sort_mapping = {
                "Training Date": "training_date",
                "Accuracy": "accuracy",
                "F1 Score": "f1_score",
                "AUC Score": "auc_score"
            }
            
            if sort_mapping[sort_by] in df.columns:
                df = df.sort_values(sort_mapping[sort_by], ascending=False)
            
            # Display performance summary
            st.subheader("📊 Performance Summary")
            
            if len(df) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    best_accuracy = df['accuracy'].max() if 'accuracy' in df.columns else 0
                    st.metric("Best Accuracy", f"{best_accuracy:.4f}" if best_accuracy > 0 else "N/A")
                
                with col2:
                    best_f1 = df['f1_score'].max() if 'f1_score' in df.columns else 0
                    st.metric("Best F1 Score", f"{best_f1:.4f}" if best_f1 > 0 else "N/A")
                
                with col3:
                    best_auc = df['auc_score'].max() if 'auc_score' in df.columns else 0
                    st.metric("Best AUC", f"{best_auc:.4f}" if best_auc > 0 else "N/A")
                
                with col4:
                    total_models = df['model_name'].nunique()
                    st.metric("Unique Models", total_models)
            
            # Model performance trends
            if len(df) > 1 and 'training_date' in df.columns:
                st.subheader("📈 Performance Trends")
                
                # Convert training_date to datetime
                df['training_date'] = pd.to_datetime(df['training_date'])
                
                # Create performance trend chart
                fig = go.Figure()
                
                for model in df['model_name'].unique():
                    model_data = df[df['model_name'] == model].sort_values('training_date')
                    
                    fig.add_trace(go.Scatter(
                        x=model_data['training_date'],
                        y=model_data['accuracy'],
                        mode='lines+markers',
                        name=model,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="Model Accuracy Over Time",
                    xaxis_title="Training Date",
                    yaxis_title="Accuracy",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display results table
            st.subheader("📋 Model Training Records")
            
            # Format datetime columns
            if 'training_date' in df.columns:
                df['training_date'] = pd.to_datetime(df['training_date']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Format numeric columns
            numeric_columns = ['accuracy', 'precision_score', 'recall', 'f1_score', 'auc_score', 'cv_mean', 'cv_std']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Model Results",
                data=csv,
                file_name=f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("No model training records found.")
    
    except Exception as e:
        st.error(f"Error retrieving model results: {str(e)}")

def manage_prediction_history():
    """Manage prediction history."""
    
    st.header("🔮 Prediction History Management")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        patient_filter = st.text_input("Filter by Patient ID:", placeholder="e.g., P001")
    
    with col2:
        model_filter = st.selectbox(
            "Filter by model:",
            ["All Models", "SVM", "ANN", "Decision_Tree", "Gaussian_NB", "Random_Forest", "Gradient_Boosting"]
        )
    
    with col3:
        limit = st.number_input("Records to display:", min_value=10, max_value=1000, value=100)
    
    try:
        # Get prediction history
        patient_id = patient_filter if patient_filter else None
        model_name = None if model_filter == "All Models" else model_filter
        
        df = st.session_state.db_manager.get_prediction_history(
            patient_id=patient_id,
            model_name=model_name,
            limit=limit
        )
        
        if not df.empty:
            st.success(f"Found {len(df)} prediction records")
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ptsd_predictions = (df['prediction'] == 1).sum()
                st.metric("PTSD Positive Predictions", ptsd_predictions)
            
            with col2:
                no_ptsd_predictions = (df['prediction'] == 0).sum()
                st.metric("PTSD Negative Predictions", no_ptsd_predictions)
            
            with col3:
                if 'probability' in df.columns:
                    avg_probability = df['probability'].mean()
                    st.metric("Avg Probability", f"{avg_probability:.3f}" if not pd.isna(avg_probability) else "N/A")
                else:
                    st.metric("Avg Probability", "N/A")
            
            with col4:
                unique_patients = df['patient_id'].nunique()
                st.metric("Unique Patients", unique_patients)
            
            # Prediction trends
            if len(df) > 1 and 'prediction_date' in df.columns:
                st.subheader("📈 Prediction Trends")
                
                # Convert prediction_date to datetime
                df['prediction_date'] = pd.to_datetime(df['prediction_date'])
                
                # Daily prediction counts
                daily_predictions = df.groupby(df['prediction_date'].dt.date).size().reset_index()
                daily_predictions.columns = ['Date', 'Count']
                
                fig = px.line(
                    daily_predictions,
                    x='Date',
                    y='Count',
                    title="Daily Prediction Count",
                    markers=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk level distribution
                if 'risk_level' in df.columns:
                    risk_counts = df['risk_level'].value_counts()
                    
                    fig_risk = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution"
                    )
                    
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            # Display predictions table
            st.subheader("📋 Prediction Records")
            
            # Format datetime columns
            if 'prediction_date' in df.columns:
                df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Format prediction labels
            if 'prediction' in df.columns:
                df['prediction_label'] = df['prediction'].map({0: 'No PTSD', 1: 'PTSD'})
            
            # Format probability
            if 'probability' in df.columns:
                df['probability'] = df['probability'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction History",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("No prediction records found.")
    
    except Exception as e:
        st.error(f"Error retrieving prediction history: {str(e)}")

def show_analytics():
    """Show advanced analytics and insights."""
    
    st.header("📊 Advanced Analytics")
    
    try:
        # Get all data for analytics
        patient_df = st.session_state.db_manager.get_patient_data(limit=1000)
        model_df = st.session_state.db_manager.get_model_results()
        prediction_df = st.session_state.db_manager.get_prediction_history(limit=1000)
        
        if patient_df.empty and model_df.empty and prediction_df.empty:
            st.info("No data available for analytics.")
            return
        
        # Patient demographics analysis
        if not patient_df.empty:
            st.subheader("👥 Patient Demographics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                if 'age' in patient_df.columns:
                    fig_age = px.histogram(
                        patient_df,
                        x='age',
                        title="Age Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                # Gender distribution
                if 'gender' in patient_df.columns:
                    gender_counts = patient_df['gender'].value_counts()
                    fig_gender = px.pie(
                        values=gender_counts.values,
                        names=gender_counts.index,
                        title="Gender Distribution"
                    )
                    st.plotly_chart(fig_gender, use_container_width=True)
            
            # PCL-5 analysis
            pcl5_columns = [col for col in patient_df.columns if 'pcl5' in col.lower()]
            if pcl5_columns:
                st.subheader("🧠 PCL-5 Analysis")
                
                # PCL-5 subscale correlations
                numeric_pcl5 = patient_df[pcl5_columns].select_dtypes(include=[np.number])
                if not numeric_pcl5.empty:
                    corr_matrix = numeric_pcl5.corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="PCL-5 Subscale Correlations",
                        color_continuous_scale='RdBu',
                        aspect="auto"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        # Model performance trends
        if not model_df.empty:
            st.subheader("🤖 Model Performance Analysis")
            
            # Model comparison
            if len(model_df) > 1:
                model_comparison = model_df.groupby('model_name').agg({
                    'accuracy': 'max',
                    'f1_score': 'max',
                    'auc_score': 'max'
                }).reset_index()
                
                fig_comparison = px.bar(
                    model_comparison,
                    x='model_name',
                    y='accuracy',
                    title="Best Accuracy by Model Type",
                    text='accuracy'
                )
                fig_comparison.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Prediction analytics
        if not prediction_df.empty:
            st.subheader("🔮 Prediction Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction distribution by model
                if 'model_name' in prediction_df.columns:
                    pred_by_model = prediction_df.groupby(['model_name', 'prediction']).size().reset_index(name='count')
                    pred_by_model['prediction_label'] = pred_by_model['prediction'].map({0: 'No PTSD', 1: 'PTSD'})
                    
                    fig_pred = px.bar(
                        pred_by_model,
                        x='model_name',
                        y='count',
                        color='prediction_label',
                        title="Predictions by Model",
                        barmode='group'
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                # Probability distribution
                if 'probability' in prediction_df.columns:
                    fig_prob = px.histogram(
                        prediction_df,
                        x='probability',
                        title="Prediction Probability Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error generating analytics: {str(e)}")

def data_export():
    """Handle data export operations."""
    
    st.header("📤 Data Export")
    
    st.markdown("Export data from the database for external analysis or backup purposes.")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Available Exports")
        
        export_patient = st.checkbox("Patient Data", value=True)
        export_models = st.checkbox("Model Results", value=True)
        export_predictions = st.checkbox("Prediction History", value=True)
    
    with col2:
        st.subheader("⚙️ Export Settings")
        
        date_range = st.date_input(
            "Date range (optional):",
            value=[],
            help="Leave empty to export all data"
        )
        
        format_type = st.selectbox(
            "Export format:",
            ["CSV", "Excel", "JSON"]
        )
    
    if st.button("📤 Generate Export", type="primary"):
        try:
            export_data = {}
            
            # Export patient data
            if export_patient:
                patient_df = st.session_state.db_manager.get_patient_data(limit=10000)
                if not patient_df.empty:
                    export_data['patient_data'] = patient_df
                    st.success(f"✅ Exported {len(patient_df)} patient records")
            
            # Export model results
            if export_models:
                model_df = st.session_state.db_manager.get_model_results()
                if not model_df.empty:
                    export_data['model_results'] = model_df
                    st.success(f"✅ Exported {len(model_df)} model results")
            
            # Export prediction history
            if export_predictions:
                prediction_df = st.session_state.db_manager.get_prediction_history(limit=10000)
                if not prediction_df.empty:
                    export_data['prediction_history'] = prediction_df
                    st.success(f"✅ Exported {len(prediction_df)} prediction records")
            
            if export_data:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if format_type == "CSV":
                    # Create ZIP file with multiple CSV files
                    import io
                    import zipfile
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for name, df in export_data.items():
                            csv_data = df.to_csv(index=False)
                            zip_file.writestr(f"{name}_{timestamp}.csv", csv_data)
                    
                    st.download_button(
                        label="📥 Download CSV Export",
                        data=zip_buffer.getvalue(),
                        file_name=f"ptsd_database_export_{timestamp}.zip",
                        mime="application/zip"
                    )
                
                elif format_type == "Excel":
                    # Create Excel file with multiple sheets
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        for name, df in export_data.items():
                            df.to_excel(writer, sheet_name=name, index=False)
                    
                    st.download_button(
                        label="📥 Download Excel Export",
                        data=excel_buffer.getvalue(),
                        file_name=f"ptsd_database_export_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                elif format_type == "JSON":
                    # Convert to JSON
                    json_data = {}
                    for name, df in export_data.items():
                        json_data[name] = df.to_dict('records')
                    
                    json_string = json.dumps(json_data, default=str, indent=2)
                    
                    st.download_button(
                        label="📥 Download JSON Export",
                        data=json_string,
                        file_name=f"ptsd_database_export_{timestamp}.json",
                        mime="application/json"
                    )
            
            else:
                st.warning("No data available for export.")
        
        except Exception as e:
            st.error(f"Error during export: {str(e)}")

if __name__ == "__main__":
    main()