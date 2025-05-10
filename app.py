import streamlit as st
import pandas as pd
import zipfile
import os
import requests

# Set the title for the Streamlit app
st.title("Diabetes Risk Prediction")
st.write("This app predicts the risk of diabetes based on the Pima Indians dataset.")

# The raw URL for the dataset in GitHub
dataset_url = "https://raw.githubusercontent.com/GOPIKA-SUSHAMA/diabetes-risk-prediction/main/archive%20(6).zip"

# Download and extract the ZIP file
response = requests.get(dataset_url)
with open("archive.zip", "wb") as file:
    file.write(response.content)

# Extract the ZIP file contents to a folder inside the current directory
with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("diabetes_data")  # Extract to 'diabetes_data' folder

# Now, load the dataset
data_path = "/workspaces/diabetes-risk-prediction/diabetes_data/diabetes.csv"  # Updated path
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
st.write(data.head())
# Preprocessing the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Separate features and target variable
X = data.drop("Outcome", axis=1)  # Features (excluding target)
y = data["Outcome"]  # Target variable (Outcome: 1 for diabetic, 0 for non-diabetic)

# Handle missing values (impute with mean)
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Scale the features (standardize)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Data Preprocessing Complete.")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the evaluation metrics in the app
st.write("Model Evaluation:")
st.write(f"Accuracy: {accuracy * 100:.2f}%")
st.write(f"Confusion Matrix: {conf_matrix}")
st.write(f"Classification Report: {class_report}")
st.header("Predict Diabetes Risk")

# Collect user input using Streamlit widgets
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=10, max_value=100, value=33)

# When the user clicks the Predict button
if st.button("Predict"):
    user_data = [[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]]
    
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("⚠️ The model predicts a **high risk of diabetes**.")
    else:
        st.success("✅ The model predicts a **low risk of diabetes**.")
