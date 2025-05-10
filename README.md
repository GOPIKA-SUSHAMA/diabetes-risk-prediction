# diabetes-risk-prediction
A web app to predict the risk of diabetes based on health metrics.
streamlit, diabetes, machine-learning, health, python, data-science
# Diabetes Risk Prediction Using Streamlit

This project uses the **Pima Indians Diabetes Dataset** to predict the risk of diabetes based on user input and visualize important features using **Streamlit**.

 **Project Structure**
diabetes-risk-prediction/
├── app.py                    # Main Streamlit app
├── diabetes_data/
│   └── diabetes.csv         # Dataset used in the app
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

### 1. **Open GitHub Codespaces** (Cloud IDE) **GitHub Codespaces** to run Streamlit app in the cloud.

### 2. **Install Required Packages**

pip install -r requirements.txt

### 3. **Run the Streamlit App**

streamlit run app.py


Click the URL shown in the terminal (usually starts with `https://`) to open the app in browser.

## Code Walkthrough (app.py)

###  Import Libraries

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io

###  App Title & Description
python
st.title("Diabetes Risk Prediction")
st.write("This app predicts the risk of diabetes based on the Pima Indians dataset.")


###  Load Dataset from Local (already extracted)
python
data_path = "diabetes_data/diabetes.csv"
data = pd.read_csv(data_path)
st.write(data.head())
```

###  Visualize Dataset

```python
st.subheader("Data Visualization")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
```

###  User Inputs for Prediction

```python
st.subheader("User Input Features")
user_input = {
    'Pregnancies': st.number_input('Pregnancies', 0, 20, 1),
    'Glucose': st.slider('Glucose', 0, 200, 120),
    'BloodPressure': st.slider('BloodPressure', 0, 122, 70),
    'SkinThickness': st.slider('SkinThickness', 0, 100, 20),
    'Insulin': st.slider('Insulin', 0, 900, 80),
    'BMI': st.slider('BMI', 0.0, 70.0, 25.0),
    'DiabetesPedigreeFunction': st.slider('DiabetesPedigreeFunction', 0.0, 2.5, 0.5),
    'Age': st.slider('Age', 0, 100, 33)
}
```

###  Load Trained Model (Assuming Future ML Implementation)

```python
# model = joblib.load('model.pkl')
# prediction = model.predict([list(user_input.values())])
# st.write("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")


## To Push to GitHub

git add .
git commit -m "Initial working diabetes prediction app"
git push origin main

##  requirements.txt Example
streamlit
pandas
matplotlib
seaborn
requests


##  Dataset Source

[Pima Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)


##  Author

[Gopika Sushama](https://github.com/GOPIKA-SUSHAMA)
