import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class PatientFlowOptimizer:
    def __init__(self):
        self.data = None
        self.model = None
        
    def load_data(self, uploaded_file):
        """Load and preprocess EHR data"""
        try:
            self.data = pd.read_csv(uploaded_file)
            # Convert date columns to datetime
            date_columns = ['appointment_date', 'check_in_time', 'check_out_time']
            for col in date_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_datetime(self.data[col])
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
                
    def analyze_patterns(self, factor1, factor2):
        """Analyze correlation between two factors"""
        try:
            if not all(col in self.data.columns for col in [factor1, factor2]):
                st.error("One or both factors not found in dataset")
                return None
            
            dtype1 = self.data[factor1].dtype
            dtype2 = self.data[factor2].dtype
            
            # Numerical vs Numerical
            if dtype1 in ['float64', 'int64'] and dtype2 in ['float64', 'int64']:
                fig = px.scatter(
                    self.data,
                    x=factor1,
                    y=factor2,
                    title=f'Relationship between {factor1} and {factor2}',
                    trendline="ols"  # Add trend line
                )
            
            # Categorical vs Numerical
            elif (dtype1 in ['object', 'category'] and dtype2 in ['float64', 'int64']):
                fig = px.box(
                    self.data,
                    x=factor1,
                    y=factor2,
                    title=f'{factor2} Distribution by {factor1}'
                )
            
            # Numerical vs Categorical
            elif (dtype1 in ['float64', 'int64'] and dtype2 in ['object', 'category']):
                # Suggest switching order
                st.warning(f"Tip: Try switching the order of factors (put {factor2} as Factor 1)")
                fig = px.box(
                    self.data,
                    x=factor2,
                    y=factor1,
                    title=f'{factor1} Distribution by {factor2}'
                )
            
            # Categorical vs Categorical
            else:
                crosstab = pd.crosstab(self.data[factor1], self.data[factor2])
                fig = px.imshow(
                    crosstab,
                    title=f'Relationship between {factor1} and {factor2}',
                    labels=dict(x=factor2, y=factor1, color="Count"),
                    aspect="auto"  # Adjust aspect ratio
                )
            
            return fig
        
        except Exception as e:
            st.error(f"Error in pattern analysis: {str(e)}")
            return None
        
    def predict_duration(self, appointment_type, patient_age, time_of_day):
        """Predict appointment duration based on various factors"""
        try:
            if self.model is None:
                # Use the actual columns from your data
                features = ['appointment_type', 'time_slot', 'department']
                
                # Prepare features for training
                X = pd.get_dummies(self.data[features])
                y = self.data['actual_duration']  # using actual_duration as target
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                self.model = RandomForestRegressor(n_estimators=100)
                self.model.fit(X_train, y_train)
                
                # Store feature importance
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Store columns for future predictions
                self.X_columns = X.columns
            
            # Prepare input for prediction
            input_data = pd.DataFrame({
                'appointment_type': [appointment_type],
                'time_slot': [time_of_day],
                'department': [self.data['department'].iloc[0]]  # using first department as default
            })
            input_encoded = pd.get_dummies(input_data)
            
            # Align input features with training features
            prediction_data = pd.DataFrame(0, index=[0], columns=self.X_columns)
            for col in input_encoded.columns:
                if col in self.X_columns:
                    prediction_data[col] = input_encoded[col]
            
            return self.model.predict(prediction_data)[0]
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.write("Data columns:", self.data.columns.tolist())
            return None
        
    def identify_no_show_risk(self):
        """Identify appointments with high no-show risk"""
        try:
            # Calculate additional risk metrics
            self.data['wait_time_risk'] = self.data['wait_time'] > self.data['wait_time'].mean()
            self.data['duration_risk'] = self.data['scheduled_duration'] > self.data['scheduled_duration'].mean()
            
            # Define risk factors based on available columns
            risk_factors = {
                'wait_time': self.data['wait_time'].quantile(0.75),  # 75th percentile
                'scheduled_duration': self.data['scheduled_duration'].quantile(0.75)
            }
            
            # Identify high-risk appointments
            high_risk = self.data[
                (self.data['wait_time'] >= risk_factors['wait_time']) |
                (self.data['scheduled_duration'] >= risk_factors['scheduled_duration']) |
                (self.data['no_show'] == 1)  # Previous no-shows
            ]
            
            # Calculate risk score (0-100)
            high_risk['risk_score'] = (
                (high_risk['wait_time'] >= risk_factors['wait_time']).astype(int) * 30 +
                (high_risk['scheduled_duration'] >= risk_factors['scheduled_duration']).astype(int) * 30 +
                (high_risk['no_show'] == 1).astype(int) * 40
            )
            
            return high_risk.sort_values('risk_score', ascending=False)
        
        except Exception as e:
            st.error(f"Error in risk analysis: {str(e)}")
            st.write("Available columns:", self.data.columns.tolist())
            return pd.DataFrame()
        
    def suggest_scheduling_template(self, department):
        """Generate optimal scheduling template based on historical data"""
        try:
            dept_data = self.data[self.data['department'] == department]
            
            # Calculate average duration by appointment type
            avg_duration = dept_data.groupby('appointment_type')['actual_duration'].mean()
            
            # Analyze time slots
            time_slot_distribution = dept_data.groupby('time_slot')['patient_id'].count()
            
            # Calculate recommended buffer based on duration variability
            duration_std = dept_data['actual_duration'].std()
            recommended_buffer = duration_std * 0.2  # 20% of standard deviation
            
            # Calculate wait time statistics
            wait_time_stats = dept_data['wait_time'].describe()
            
            # Calculate utilization by appointment type
            utilization = (dept_data.groupby('appointment_type')['actual_duration'].sum() / 
                          dept_data.groupby('appointment_type')['scheduled_duration'].sum() * 100)
            
            return {
                'average_duration': avg_duration,
                'time_slot_distribution': time_slot_distribution,
                'recommended_buffer': recommended_buffer,
                'wait_time_stats': wait_time_stats,
                'utilization': utilization
            }
        except Exception as e:
            st.error(f"Error generating template: {str(e)}")
            st.write("Available columns:", self.data.columns.tolist())
            return None

# Streamlit app
st.set_page_config(page_title="Patient Flow Optimizer", layout="wide")

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = PatientFlowOptimizer()

# Sidebar navigation
page = st.sidebar.selectbox(
    'Select Analysis Page',
    ['Data Upload', 'Pattern Analysis', 'Duration Prediction', 'No-Show Risk', 'Department Analytics']
)

# Data Upload Page
if page == 'Data Upload':
    st.title("Patient Flow Data Upload")
    uploaded_file = st.file_uploader("Upload EHR Data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        if st.session_state.optimizer.load_data(uploaded_file):
            st.success("Data loaded successfully!")
            st.write("Data Preview:")
            st.write(st.session_state.optimizer.data.head())
        
# Pattern Analysis Page
elif page == 'Pattern Analysis':
    st.title("Pattern Analysis")
    
    if st.session_state.optimizer.data is None:
        st.warning("Please upload data first!")
    else:
        # Add guidance about factor selection
        st.info("""
        ### 📊 Factor Selection Guide
        
        The visualization type depends on the data types of selected factors:
        - **Categorical vs Numerical**: Shows box plots (Factor 1 should be categorical)
        - **Categorical vs Categorical**: Shows heatmap of relationships
        - **Numerical vs Numerical**: Shows scatter plot
        
        Try switching the order of factors if the visualization isn't optimal!
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            factor1 = st.selectbox(
                "Select Factor 1",
                st.session_state.optimizer.data.columns,
                help="For optimal visualization, categorical factors work best as Factor 1"
            )
            # Add data type info
            dtype1 = st.session_state.optimizer.data[factor1].dtype
            st.caption(f"Data type: {dtype1}")
            
        with col2:
            factor2 = st.selectbox(
                "Select Factor 2",
                st.session_state.optimizer.data.columns,
                help="For optimal visualization, numerical factors work best as Factor 2"
            )
            # Add data type info
            dtype2 = st.session_state.optimizer.data[factor2].dtype
            st.caption(f"Data type: {dtype2}")
        
        # Add suggestion about factor ordering
        if (dtype1 in ['float64', 'int64']) and (dtype2 in ['object', 'category']):
            st.warning("💡 Tip: Try switching the order of factors for better visualization!")
            
        if st.button("Analyze Patterns"):
            fig = st.session_state.optimizer.analyze_patterns(factor1, factor2)
            if fig is not None:
                st.plotly_chart(fig)
                
                # Add interpretation guidance
                st.subheader("Interpretation Guide")
                if dtype1 in ['object', 'category'] and dtype2 in ['float64', 'int64']:
                    st.markdown(f"""
                    This box plot shows the distribution of {factor2} for each {factor1} category:
                    - The box shows the middle 50% of values
                    - The line in the box is the median
                    - The whiskers show the range of typical values
                    - Points beyond the whiskers are potential outliers
                    """)
                elif dtype1 in ['object', 'category'] and dtype2 in ['object', 'category']:
                    st.markdown(f"""
                    This heatmap shows the relationship between {factor1} and {factor2}:
                    - Darker colors indicate more frequent combinations
                    - Lighter colors indicate less frequent combinations
                    """)
                else:
                    st.markdown(f"""
                    This visualization shows the relationship between {factor1} and {factor2}.
                    Consider switching the order of factors if the pattern isn't clear.
                    """)

# Duration Prediction Page
elif page == 'Duration Prediction':
    st.title("Appointment Duration Prediction")
    
    if st.session_state.optimizer.data is None:
        st.warning("Please upload data first!")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            apt_type = st.selectbox(
                "Appointment Type", 
                st.session_state.optimizer.data['appointment_type'].unique(),
                help="Select the type of appointment"
            )
        
        with col2:
            department = st.selectbox(
                "Department",
                st.session_state.optimizer.data['department'].unique(),
                help="Select the department"
            )
        
        with col3:
            time_slot = st.selectbox(
                "Time Slot", 
                st.session_state.optimizer.data['time_slot'].unique(),
                help="Select the time of day"
            )
        
        if st.button("Predict Duration"):
            duration = st.session_state.optimizer.predict_duration(apt_type, department, time_slot)
            if duration is not None:
                st.success(f"Predicted duration: {duration:.2f} minutes")
                
                # Show feature importance
                if hasattr(st.session_state.optimizer, 'feature_importance'):
                    st.subheader("Feature Importance")
                    fig = px.bar(
                        st.session_state.optimizer.feature_importance,
                        x='feature', 
                        y='importance',
                        title="Feature Importance in Prediction Model"
                    )
                    st.plotly_chart(fig)

                    # Show additional statistics
                    st.subheader("Historical Statistics")
                    dept_stats = st.session_state.optimizer.data[
                        (st.session_state.optimizer.data['appointment_type'] == apt_type) &
                        (st.session_state.optimizer.data['department'] == department)
                    ]['actual_duration'].describe()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Duration", f"{dept_stats['mean']:.1f} min")
                        st.metric("Maximum Duration", f"{dept_stats['max']:.1f} min")
                    with col2:
                        st.metric("Minimum Duration", f"{dept_stats['min']:.1f} min")
                        st.metric("Standard Deviation", f"{dept_stats['std']:.1f} min")

# No-Show Risk Page
elif page == 'No-Show Risk':
    st.title("No-Show Risk Analysis")
    
    if st.session_state.optimizer.data is None:
        st.warning("Please upload data first!")
    else:
        high_risk = st.session_state.optimizer.identify_no_show_risk()
        
        if not high_risk.empty:
            # Summary metrics
            total_appointments = len(st.session_state.optimizer.data)
            high_risk_count = len(high_risk)
            risk_percentage = (high_risk_count / total_appointments) * 100
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Appointments", total_appointments)
            with col2:
                st.metric("High Risk Appointments", high_risk_count)
            with col3:
                st.metric("Risk Percentage", f"{risk_percentage:.1f}%")
            
            # Risk distribution
            st.subheader("Risk Score Distribution")
            fig = px.histogram(high_risk, x='risk_score', 
                             title="Distribution of Risk Scores",
                             labels={'risk_score': 'Risk Score', 'count': 'Number of Appointments'})
            st.plotly_chart(fig)
            
            # High risk appointments table
            st.subheader("High Risk Appointments")
            display_columns = ['appointment_id', 'date', 'appointment_type', 
                             'department', 'wait_time', 'risk_score']
            st.dataframe(high_risk[display_columns])
            
            # Risk factors analysis
            st.subheader("Risk Factors Analysis")
            risk_factors = {
                'Long Wait Time': len(high_risk[high_risk['wait_time_risk']]),
                'Long Duration': len(high_risk[high_risk['duration_risk']]),
                'Previous No-Show': len(high_risk[high_risk['no_show'] == 1])
            }
            
            fig2 = px.bar(x=list(risk_factors.keys()), y=list(risk_factors.values()),
                         title="Risk Factors Distribution",
                         labels={'x': 'Risk Factor', 'y': 'Number of Appointments'})
            st.plotly_chart(fig2)
            
            # Download high-risk appointments
            csv = high_risk[display_columns].to_csv(index=False)
            st.download_button(
                label="Download High Risk Appointments",
                data=csv,
                file_name="high_risk_appointments.csv",
                mime="text/csv"
            )
        else:
            st.info("No high-risk appointments identified.")

# Department Analytics Page
elif page == 'Department Analytics':
    st.title("Department Performance Analytics")
    st.markdown("""
    This dashboard provides key performance metrics and insights by department, 
    including appointment durations, wait times, and resource utilization patterns.
    """)
    
    if st.session_state.optimizer.data is None:
        st.warning("Please upload data first!")
    else:
        department = st.selectbox("Select Department", 
                                st.session_state.optimizer.data['department'].unique())
        
        if st.button("Analyze"):
            template = st.session_state.optimizer.suggest_scheduling_template(department)
            
            if template is not None:
                # Display average duration
                st.subheader("Average Duration by Appointment Type")
                fig1 = px.bar(
                    x=template['average_duration'].index,
                    y=template['average_duration'].values,
                    labels={'x': 'Appointment Type', 'y': 'Duration (minutes)'},
                    title="Average Appointment Duration"
                )
                st.plotly_chart(fig1)
                
                # Display time slot distribution
                st.subheader("Patient Volume by Time Slot")
                fig2 = px.bar(
                    x=template['time_slot_distribution'].index,
                    y=template['time_slot_distribution'].values,
                    labels={'x': 'Time Slot', 'y': 'Number of Patients'},
                    title="Patient Volume Distribution"
                )
                st.plotly_chart(fig2)
                
                # Display utilization
                st.subheader("Resource Utilization by Appointment Type")
                fig3 = px.bar(
                    x=template['utilization'].index,
                    y=template['utilization'].values,
                    labels={'x': 'Appointment Type', 'y': 'Utilization (%)'},
                    title="Resource Utilization"
                )
                st.plotly_chart(fig3)
                
                # Display recommendations
                st.subheader("Scheduling Recommendations")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Recommended Buffer Time",
                        f"{template['recommended_buffer']:.1f} min"
                    )
                    st.metric(
                        "Average Wait Time",
                        f"{template['wait_time_stats']['mean']:.1f} min"
                    )
                
                with col2:
                    st.metric(
                        "Max Wait Time",
                        f"{template['wait_time_stats']['max']:.1f} min"
                    )
                    st.metric(
                        "Wait Time Std Dev",
                        f"{template['wait_time_stats']['std']:.1f} min"
                    )
                
                # Additional recommendations
                st.subheader("Optimization Suggestions")
                busiest_slot = template['time_slot_distribution'].idxmax()
                longest_apt = template['average_duration'].idxmax()
                
                st.markdown(f"""
                ### Key Findings:
                - Busiest time slot: **{busiest_slot}**
                - Longest appointment type: **{longest_apt}**
                
                ### Recommendations:
                1. Add {template['recommended_buffer']:.1f} minutes buffer between appointments
                2. Schedule {longest_apt} appointments during less busy slots
                3. Consider additional resources during {busiest_slot} slot
                4. Monitor and adjust scheduling based on wait time patterns
                """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This tool helps optimize patient flow by:
- Analyzing appointment patterns
- Predicting appointment durations
- Identifying no-show risks
- Generating scheduling templates
""")
