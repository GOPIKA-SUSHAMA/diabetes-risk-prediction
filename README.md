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



pip install -r requirements.txt



streamlit run app.py




import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io

python
st.title("Diabetes Risk Prediction")
st.write("This app predicts the risk of diabetes based on the Pima Indians dataset.")



python
data_path = "diabetes_data/diabetes.csv"
data = pd.read_csv(data_path)
st.write(data.head())
```

st.subheader("Data Visualization")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
```


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


# model = joblib.load('model.pkl')
# prediction = model.predict([list(user_input.values())])
# st.write("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
=

git add .
git commit -m "Initial working diabetes prediction app"
git push origin main


streamlit
pandas
matplotlib
seaborn
requests


##  Dataset Source

[Pima Indians Diabetes Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)


##  Author

[Gopika Sushama](https://github.com/GOPIKA-SUSHAMA)
