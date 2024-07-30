import streamlit as st
import requests
import json

st.title("Predicting Iris Class")
st.subheader("with AI Model")

sepal_length = st.text_area("sepal length")
sepal_width = st.text_area("sepal width")
petal_length = st.text_area("petal length")
petal_width = st.text_area("petal width")

info_dict = {"sepal_length": sepal_length, 
             "sepal_width": sepal_width, 
             "petal_length": petal_length,
             "petal_width": petal_width}

result = None
if st.button("Predict"):
    result = requests.post(
        url="http://127.0.0.1:8000/predict",
        data=json.dumps(info_dict),
        verify=False)
    if result.status_code == 200:
        response_data = result.json()  # JSON 데이터를 추출
        result = response_data.get("iris_class")  # 'class' 키의 값을 추출
    else:
        result = f"Error: {result.status_code}"

st.subheader(f"Iris class: {result}")