import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm  # Ensure lightgbm is imported

# Set page config
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        st.write("Loading model...")
        model = joblib.load('models/best_wine_quality_model.joblib')
        st.write("Model loaded successfully!")
        
        st.write("Loading scaler...")
        scaler = joblib.load('models/scaler.joblib')
        st.write("Scaler loaded successfully!")
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {str(e)}")
        raise e

model, scaler = load_model()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('winequality.csv')
    X = df.drop('quality', axis=1)
    y = df['quality'] - df['quality'].min()
    return df, X, y

df, X, y = load_data()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analytics", "About"])

# Home Page
if page == "Home":
    st.title("🍷 Wine Quality Prediction")
    
    # Input features in sidebar
    st.sidebar.header("Input Features")
    features = {}
    for col in X.columns:
        features[col] = st.sidebar.number_input(
            col, 
            value=float(X[col].mean()),
            step=0.1
        )
    
    # Prediction
    if st.sidebar.button("Predict Quality"):
        try:
            input_data = pd.DataFrame([features])
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0] + df['quality'].min()
            
            st.success(f"Predicted Wine Quality: {prediction}/10")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Analytics Page
elif page == "Analytics":
    st.title("📈 Model Analytics")
    
    # Generate predictions
    y_pred = model.predict(scaler.transform(X))
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Accuracy
    st.subheader("Model Accuracy")
    accuracy = accuracy_score(y, y_pred)
    st.metric("Accuracy", f"{accuracy:.2%}")
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
        st.pyplot(fig)

# About Page
elif page == "About":
    st.title("About the Project")
    
    st.markdown("""
    ## 🍷 Wine Quality Prediction Project
    
    ### Project Overview
    This application predicts wine quality based on physicochemical properties using machine learning.
    
    ### Key Features:
    - Interactive quality prediction
    - Model performance analytics
    - Feature importance visualization
    
    ### Technical Details:
    - **Model Type**: LightGBM Classifier
    - **Accuracy**: {:.2f}%
    - **Dataset**: Wine Quality Dataset ({} samples)
    """.format(accuracy_score(y, model.predict(scaler.transform(X))) * 100, len(df)))
    
    st.markdown("---")
    st.write("Built with ❤️ using Streamlit")
