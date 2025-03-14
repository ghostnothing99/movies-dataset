import streamlit as st
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model and scaler
@st.cache_resource
def load_model():
    model_path = 'models/best_wine_quality_model.joblib'
    scaler_path = 'models/scaler.joblib'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        st.success("Model and scaler loaded successfully!")
        return model, scaler
    else:
        st.error("Model or scaler file not found. Please ensure the files exist.")
        st.stop()

model, scaler = load_model()

# Load dataset
@st.cache_data
def load_data():
    data_path = 'D:/My/Wine/data/winequality.csv'
    if not os.path.exists(data_path):
        st.error(f"Dataset file not found at: {data_path}")
        st.stop()
    
    df = pd.read_csv(data_path)
    X = df.drop('quality', axis=1)
    y = df['quality'] - df['quality'].min()  # Normalize labels to start from 0

    # Ensure feature alignment
    if hasattr(scaler, 'feature_names_in_'):
        required_features = list(scaler.feature_names_in_)
        missing_features = [feat for feat in required_features if feat not in X.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            st.stop()
        X = X[required_features]

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
    input_features = {}
    for feature in scaler.feature_names_in_:
        input_features[feature] = st.sidebar.number_input(
            feature,
            value=float(X[feature].mean()),
            step=0.1
        )

    # Prediction
    if st.sidebar.button("Predict Quality"):
        try:
            input_df = pd.DataFrame([input_features])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0] + df['quality'].min()
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
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Accuracy
    st.subheader("Model Accuracy")
    accuracy = accuracy_score(y, y_pred)
    st.metric("Accuracy", f"{accuracy:.2%}")

    # ROC Curve (if applicable)
    if hasattr(model, "predict_proba") and len(np.unique(y)) > 2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(np.unique(y))):
            y_true = (y == i).astype(int)
            y_probs = model.predict_proba(scaler.transform(X))[:, i]
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    # Precision-Recall Curve (if applicable)
    if hasattr(model, "predict_proba"):
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(np.unique(y))):
            y_true = (y == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true, model.predict_proba(scaler.transform(X))[:, i])
            ax.plot(recall, precision, label=f'Class {i}')
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        st.pyplot(fig)

# About Page
elif page == "About":
    st.title("About the Project")

    # Evaluate the best model
    y_pred = model.predict(scaler.transform(X))
    test_accuracy = accuracy_score(y, y_pred)

    st.markdown("""
    ## 🍷 Wine Quality Prediction Project
    
    ### Project Overview
    This application predicts wine quality based on physicochemical properties using machine learning.
    
    ### Key Features:
    - Interactive quality prediction
    - Model performance analytics
    - Feature importance visualization
    
    ### Technical Details:
    - **Model Type**: Random Forest Classifier
    - **Accuracy**: {:.2f}%
    - **Dataset**: Wine Quality Dataset ({} samples)
    """.format(test_accuracy * 100, len(df)))

    st.markdown("---")
    st.write("Built with ❤️ using Streamlit")
