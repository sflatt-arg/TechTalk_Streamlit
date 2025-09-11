import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')
from supabase_client import supabase

# Page configuration
st.header("🚴 Bike Rental Late Return Predictor")
st.markdown("---")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'contextual_cols' not in st.session_state:
    st.session_state.contextual_cols = []
if 'df_train' not in st.session_state:
    st.session_state.df_train = None

def load_and_prepare_data(df):
    """Load and prepare the dataset for training with flexible column handling"""
    
    # Check and handle timestamp column (flexible naming)
    timestamp_cols = ['start_ts', 'timestamp', 'created_at', 'start_time', 'ride_start']
    timestamp_col = None
    
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col:
        df['start_ts'] = pd.to_datetime(df[timestamp_col])
    else:
        # Create a dummy timestamp if none exists
        df['start_ts'] = pd.Timestamp.now()
        st.warning("No timestamp column found. Using current time as default.")
    
    # Extract hour from timestamp
    df['hour'] = df['start_ts'].dt.hour
    
    # Features to use for training (excluding contextual features that won't be available at prediction time)
    feature_cols = [
        'start_station', 'weekday', 'is_weekend', 'is_holiday', 'month',
        'planned_duration_min', 'weather', 'temp_bucket', 'wind_bucket', 
        'event_nearby', 'hour'
    ]
    
    # Contextual features (available in training but need to be estimated/retrieved at prediction time)
    contextual_cols = ['station_congestion_base', 'network_congestion_index', 'user_tardiness_propensity']
    
    # Check if required columns exist
    missing_cols = []
    for col in feature_cols + contextual_cols + ['late_return']:
        if col not in df.columns and col != 'hour':  # hour is derived
            missing_cols.append(col)
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())
        return None, None, None
    
    return df, feature_cols, contextual_cols

def preprocess_features(df, feature_cols, contextual_cols, encoders=None, scaler=None, is_training=True):
    """Preprocess features for training or prediction"""
    df_processed = df.copy()

    # Ensure numerical columns are properly typed
    numerical_cols = ['weekday', 'is_weekend', 'is_holiday', 'month', 'planned_duration_min', 'event_nearby', 'hour']
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Initialize encoders if training
    if is_training:
        encoders = {}
        scaler = StandardScaler()
    
    # Encode categorical variables
    categorical_cols = ['start_station', 'weather', 'temp_bucket', 'wind_bucket']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            if is_training:
                le = LabelEncoder()
                df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
                encoders[col] = le
            else:
                if col in encoders:
                    # Handle unseen categories
                    le = encoders[col]
                    df_processed[f'{col}_encoded'] = df_processed[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
                else:
                    df_processed[f'{col}_encoded'] = 0
    
    # Prepare feature matrix
    encoded_categorical = [f'{col}_encoded' for col in categorical_cols if col in df_processed.columns]
    numerical_cols = ['weekday', 'is_weekend', 'is_holiday', 'month', 'planned_duration_min', 'event_nearby', 'hour']
    
    # Add contextual features if available
    available_contextual = [col for col in contextual_cols if col in df_processed.columns]
    
    all_features = numerical_cols + encoded_categorical + available_contextual
    
    # Filter features that actually exist
    existing_features = [col for col in all_features if col in df_processed.columns]
    X = df_processed[existing_features]
    
    # Scale numerical features
    numerical_indices = [i for i, col in enumerate(existing_features) if col in numerical_cols + available_contextual]
    
    if is_training:
        X_scaled = X.copy()
        if numerical_indices:
            X_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X.iloc[:, numerical_indices])
    else:
        X_scaled = X.copy()
        if numerical_indices and scaler is not None:
            X_scaled.iloc[:, numerical_indices] = scaler.transform(X.iloc[:, numerical_indices])
    
    return X_scaled, encoders, scaler


def estimate_contextual_features(input_data, df_train):
    """Estimate contextual features based on historical data"""
    station = input_data['start_station']
    hour = input_data['hour']
    is_weekend = input_data['is_weekend']
    
    # Estimate station congestion base (average for the station)
    if 'station_congestion_base' in df_train.columns:
        # CONVERT TO NUMERIC FIRST
        df_train['station_congestion_base'] = pd.to_numeric(df_train['station_congestion_base'], errors='coerce')
        
        station_filter = df_train['start_station'] == station
        if station_filter.sum() > 0:
            station_data = df_train[station_filter]['station_congestion_base']
            station_cong = float(station_data.mean())
        else:
            station_cong = float(df_train['station_congestion_base'].mean())
        
        if pd.isna(station_cong):
            station_cong = 0.5
    else:
        station_cong = 0.5
    
    # Estimate network congestion (based on hour and weekend)  
    if 'network_congestion_index' in df_train.columns:
        # CONVERT TO NUMERIC FIRST
        df_train['network_congestion_index'] = pd.to_numeric(df_train['network_congestion_index'], errors='coerce')
        
        if 'hour' not in df_train.columns:
            df_train['hour'] = pd.to_datetime(df_train['start_ts']).dt.hour
        
        similar_conditions = df_train[
            (df_train['hour'] == hour) & 
            (df_train['is_weekend'] == is_weekend)
        ]['network_congestion_index']
        
        if len(similar_conditions) > 0:
            network_cong = float(similar_conditions.mean())
        else:
            network_cong = float(df_train['network_congestion_index'].mean())
        
        if pd.isna(network_cong):
            network_cong = 0.3
    else:
        network_cong = 0.3
    
    # Use average user propensity
    if 'user_tardiness_propensity' in df_train.columns:
        # CONVERT TO NUMERIC FIRST
        df_train['user_tardiness_propensity'] = pd.to_numeric(df_train['user_tardiness_propensity'], errors='coerce')
        
        user_prop = float(df_train['user_tardiness_propensity'].mean())
        
        if pd.isna(user_prop):
            user_prop = 0.2
    else:
        user_prop = 0.2
    
    return station_cong, network_cong, user_prop

# Main app layout
tab1, tab2 = st.tabs(["📊 Training Section", "🎯 Prediction Section"])

with tab1:
    st.header("Model Training and Evaluation")
    
    # Data loading
    if st.button("🔄 Load Data from Database"):
        with st.spinner("Loading data from Supabase..."):
            try:
                all_data = []
                page_size = 5000
                start = 0
                
                while True:
                    response = supabase.table("late_return").select("*").range(start, start + page_size - 1).execute()
                    batch_data = response.data
                    
                    if not batch_data:  # No more data
                        break
                        
                    all_data.extend(batch_data)
                    start += page_size
                    
                    if len(batch_data) < page_size:  # Last batch
                        break
                
                df_raw = pd.DataFrame(all_data)
                st.success(f"Loaded {len(df_raw)} total rows!")
                
                result = load_and_prepare_data(df_raw)
                if result[0] is not None:
                    df, feature_cols, contextual_cols = result
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.feature_cols = feature_cols
                    st.session_state.contextual_cols = contextual_cols
                    st.session_state.data_loaded = True
                    
                    st.success("Data prepared successfully!")
                else:
                    st.error("Failed to prepare data. Check column names.")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.write("Make sure your Supabase connection is working and the table exists.")

    # Check if data is loaded from session state
    if st.session_state.get('data_loaded', False):
        df = st.session_state.df
        feature_cols = st.session_state.feature_cols
        contextual_cols = st.session_state.contextual_cols

        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Display basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rides", len(df))
        with col2:
            if 'late_return' in df.columns:
                st.metric("Late Return Rate", f"{df['late_return'].mean():.2%}")
            else:
                st.metric("Late Return Rate", "N/A")
        with col3:
            if 'start_station' in df.columns:
                st.metric("Number of  Stations", df['start_station'].nunique())
            else:
                st.metric("Number of  Stations", "N/A")
        
        # Show data preview
        if st.checkbox("Show data preview"):
            st.dataframe(df.head())
        
        # Check if we have the target variable
        if 'late_return' not in df.columns:
            st.error("Missing 'late_return' target column. Cannot proceed with training.")
        else:
            # Model selection
            st.subheader("Model Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox("Select Model Type", 
                                        ["Random Forest", "Logistic Regression"])
            
            with col2:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            
            # Train model button
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare features
                        X, encoders, scaler = preprocess_features(
                            df, feature_cols, contextual_cols, is_training=True
                        )
                        y = df['late_return']
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        # Train model
                        if model_type == "Random Forest":
                            model = RandomForestClassifier(
                                n_estimators=100, 
                                max_depth=10, 
                                random_state=42,
                                class_weight='balanced'
                            )
                        else:
                            model = LogisticRegression(
                                random_state=42, 
                                class_weight='balanced',
                                max_iter=1000
                            )
                        
                        model.fit(X_train, y_train)
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.encoders = encoders
                        st.session_state.scaler = scaler
                        st.session_state.feature_names = X.columns.tolist()
                        st.session_state.model_trained = True
                        st.session_state.df_train = df
                        
                        # Predictions
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        # Metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        auc = roc_auc_score(y_test, y_pred_proba)
                        
                        st.success("Model trained successfully!")
                        
                        # Display metrics
                        st.subheader("📈 Model Performance")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.3f}")
                        with col2:
                            st.metric("Precision", f"{precision:.3f}")
                        with col3:
                            st.metric("Recall", f"{recall:.3f}")
                        with col4:
                            st.metric("F1-Score", f"{f1:.3f}")
                        with col5:
                            st.metric("AUC-ROC", f"{auc:.3f}")
                        
                        # Confusion Matrix
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig_cm = px.imshow(cm, 
                                         text_auto=True, 
                                         aspect="auto",
                                         title="Confusion Matrix",
                                         labels=dict(x="Predicted", y="Actual"))
                        fig_cm.update_xaxes(tickvals=[0, 1], ticktext=['On Time', 'Late'])
                        fig_cm.update_yaxes(tickvals=[0, 1], ticktext=['On Time', 'Late'])
                        
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # Feature Importance (for Random Forest)
                        if model_type == "Random Forest":
                            feature_importance = pd.DataFrame({
                                'feature': X.columns,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            fig_importance = px.bar(
                                feature_importance.head(10), 
                                x='importance', 
                                y='feature',
                                title="Top 10 Feature Importance",
                                orientation='h'
                            )
                            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Classification Report
                        with st.expander("📋 Detailed Classification Report"):
                            st.text(classification_report(y_test, y_pred, target_names=['On Time', 'Late']))
                            
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        st.write("Check that your data has the required columns and format.")

with tab2:
    st.header("Late Return Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the Training Section.")
    else:
        st.success("✅ Model is ready for predictions!")
        
        # Input form
        st.subheader("Enter Ride Details")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                start_station = st.selectbox(
                    "Start Station", 
                    options=['S1', 'S2', 'S3', 'S4', 'S5']
                )
                
                start_date = st.date_input("Start Date", value=datetime.now().date())
                start_time = st.time_input("Start Time", value=time(9, 0))
                
                weekday = st.selectbox(
                    "Day of Week", 
                    options=[0, 1, 2, 3, 4, 5, 6],
                    format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
                )
                
                is_weekend = st.checkbox("Weekend")
                is_holiday = st.checkbox("Holiday")
                
            with col2:
                month = st.selectbox("Month", options=list(range(1, 13)))
                
                planned_duration_min = st.slider(
                    "Planned Duration (minutes)", 
                    min_value=5, 
                    max_value=120, 
                    value=30
                )
                
                weather = st.selectbox(
                    "Weather", 
                    options=['sun', 'clouds', 'rain', 'wind', 'snow']
                )
                
                temp_bucket = st.selectbox(
                    "Temperature", 
                    options=['cold', 'mild', 'warm']
                )
                
                wind_bucket = st.selectbox(
                    "Wind Conditions", 
                    options=['calm', 'breezy', 'windy']
                )
                
                event_nearby = st.checkbox("Event Nearby")
            
            # Predict button
            submitted = st.form_submit_button("🔮 Predict Late Return Probability", type="primary")
            
            if submitted:
                # Check if we have training data available
                if st.session_state.df_train is None:
                    st.error("No training data available. Please train a model first.")
                else:
                    try:
                        st.write("🔍 **Step 1: Creating input data**")
                        # Prepare input data
                        input_data = {
                            'start_station': start_station,
                            'weekday': int(weekday),  # Ensure int conversion
                            'is_weekend': int(is_weekend),
                            'is_holiday': int(is_holiday),
                            'month': int(month),  # Ensure int conversion
                            'planned_duration_min': int(planned_duration_min),  # Ensure int conversion
                            'weather': weather,
                            'temp_bucket': temp_bucket,
                            'wind_bucket': wind_bucket,
                            'event_nearby': int(event_nearby),
                            'hour': int(start_time.hour)  # Ensure int conversion
                        }
                    
                        # Estimate contextual features
                        station_cong, network_cong, user_prop = estimate_contextual_features(
                            input_data, st.session_state.df_train
                        )
                        
                        # Check for NaN or infinite values
                        if pd.isna(station_cong) or pd.isna(network_cong) or pd.isna(user_prop):
                            st.error("Found NaN values in contextual features!")
                            st.stop()
                        
                        if not np.isfinite(station_cong) or not np.isfinite(network_cong) or not np.isfinite(user_prop):
                            st.error("Found infinite values in contextual features!")
                            st.stop()
                        
                        input_data.update({
                            'station_congestion_base': float(station_cong),
                            'network_congestion_index': float(network_cong),
                            'user_tardiness_propensity': float(user_prop)
                        })
                        st.success("✅ Step 2 completed")
                        
                        st.write("🔍 **Step 3: Creating DataFrame**")
                        # Create DataFrame
                        input_df = pd.DataFrame([input_data])
                        
                        # Check for any problematic values in the DataFrame
                        for col in input_df.columns:
                            val = input_df[col].iloc[0]
                            if pd.isna(val):
                                st.error(f"Found NaN in column {col}")
                                st.stop()
                            if isinstance(val, (int, float)) and not np.isfinite(val):
                                st.error(f"Found infinite value in column {col}: {val}")
                                st.stop()
                        

                        # Preprocess
                        X_pred, _, _ = preprocess_features(
                            input_df, 
                            st.session_state.feature_cols,
                            st.session_state.contextual_cols,
                            encoders=st.session_state.encoders,
                            scaler=st.session_state.scaler,
                            is_training=False
                        )
                        
                        X_pred = X_pred.reindex(columns=st.session_state.feature_names, fill_value=0)
                        
                        # Final check before prediction
                        for col in X_pred.columns:
                            val = X_pred[col].iloc[0]
                            if pd.isna(val):
                                st.error(f"Found NaN in final column {col}")
                                st.stop()
                            if isinstance(val, (int, float)) and not np.isfinite(val):
                                st.error(f"Found infinite value in final column {col}: {val}")
                                st.stop()
                        
                        # Make prediction
                        probability = st.session_state.model.predict_proba(X_pred)[0, 1]
                        prediction = "Late" if probability > 0.5 else "On Time"
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction", prediction)
                        with col2:
                            st.metric("Late Return Probability", f"{probability:.1%}")
                        
                        # Probability gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Late Return Probability (%)"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Risk factors
                        with st.expander("📊 Risk Factor Analysis"):
                            st.write("**Estimated Contextual Factors:**")
                            st.write(f"- Station Congestion: {station_cong:.2f}")
                            st.write(f"- Network Congestion: {network_cong:.2f}")
                            st.write(f"- User Tardiness Propensity: {user_prop:.2f}")
                            
                            risk_factors = []
                            if weather in ['rain', 'snow']:
                                risk_factors.append(f"Adverse weather ({weather})")
                            if temp_bucket == 'cold':
                                risk_factors.append("Cold temperature")
                            if wind_bucket == 'windy':
                                risk_factors.append("Windy conditions")
                            if event_nearby:
                                risk_factors.append("Event nearby")
                            if planned_duration_min > 60:
                                risk_factors.append("Long planned duration")
                            if is_weekend:
                                risk_factors.append("Weekend timing")
                            
                            if risk_factors:
                                st.write("**Contributing Risk Factors:**")
                                for factor in risk_factors:
                                    st.write(f"- {factor}")
                            else:
                                st.write("**No major risk factors identified.**")
                                
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
                        st.write("**Full error traceback:**")
                        import traceback
                        st.code(traceback.format_exc())

# Sidebar with app info
with st.sidebar:
    st.markdown("### 📱 App Information")
    st.markdown("""
    This application predicts the probability of late bike returns based on:
    - Weather conditions
    - Time factors  
    - Station characteristics
    - Trip planning details
    
    **How to use:**
    1. Load your training data in the Training section
    2. Train your preferred model
    3. Use the Prediction section to forecast late returns
    """)
    
    if st.session_state.model_trained:
        st.success("✅ Model Ready")
        # Download model option
        if st.button("💾 Download Model"):
            model_data = {
                'model': st.session_state.model,
                'encoders': st.session_state.encoders,
                'scaler': st.session_state.scaler,
                'feature_names': st.session_state.feature_names
            }
            st.download_button(
                label="Download Trained Model",
                data=pickle.dumps(model_data),
                file_name="bike_return_model.pkl",
                mime="application/octet-stream"
            )
    else:
        st.warning("⏳ Model Not Trained")