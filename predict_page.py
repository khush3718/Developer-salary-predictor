import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary Estimator Tool")
    st.write("""### Enter Your Information for an Accurate Estimate""")
    st.write("""In order to receive an accurate estimate of the salary, we kindly request that you provide us with some necessary information, including your country of residence, highest level of education completed, and years of experience in your field. """)

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden"
    )

    education = (
        'Bachelor’s degree', 
         'Master’s degree', 
         'Less than a Bachelors',
         'Post grad'
         )
    
    country = st.selectbox("Country",countries)
    education = st.selectbox("Education Level", education)
    
    experience = st.slider("Years of Experience", 0, 50, 3)
    
    btn = st.button("Calculate Salary")
    
    if btn:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:0.2f}")
        

