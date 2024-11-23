import tensorflow as tf
import pandas as pd
import numpy as np
import kagglehub
import os
import datetime
import streamlit as st
from dateutil.relativedelta import relativedelta


@st.cache_data
def get_mean_std():
    try:
        path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
        dataset = pd.read_csv(os.path.join(path, "cardio_train.csv"), delimiter=";")
    except:
        dataset = pd.read_csv("cardio_train.csv", delimiter=";")

    predictor_numerical =  dataset[["age", "height", "weight", "ap_hi", "ap_lo"]]
    predictor_numerical['age'] = predictor_numerical['age']/365
    mean = predictor_numerical.mean()
    std = predictor_numerical.std()

    return mean, std

@st.cache_resource
def get_model():
    return tf.keras.models.load_model("cardiovascular_nn_model.keras")

mean, std = get_mean_std()

model = get_model()

predictor_test = pd.DataFrame(0, index=[0], 
                              columns=['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'gender_1', 'gender_2',
                                       'cholesterol_1', 'cholesterol_2', 'cholesterol_3', 'gluc_1', 'gluc_2',
                                       'gluc_3', 'smoke_0', 'smoke_1', 'alco_0', 'alco_1', 'active_0','active_1']).astype(float)

predictor_numerical_test = pd.DataFrame(columns=["age", "height", "weight", "ap_hi", "ap_lo"]).astype(float)
predictor_nominal_test = pd.DataFrame(columns=["gender", "cholesterol", "gluc", "smoke", "alco", "active"]).astype(str)

three_level_id = {"Normal":"1", "Above normal":"2", "Well above normal":"3"}
two_level_id = {"Yes":"1", "No":"0"}
gender_id = {"Women":"1", "Men":"2"}

st.title("Cardiovascular Disease Detection")
st.write("This is description.")

birth = st.date_input("When's your birthday", datetime.date(2000, 1, 1))
height = st.slider("How tall are you?", 0, 200, 180)
weigth = st.slider("How much do you weight?", 0, 200, 75)
ap_hi = st.slider("What's your systolic blood pressure?", 0, 200, 120)
ap_lo = st.slider("What's your diastolic blood pressure?", 0, 200, 80)

cholesterol = st.selectbox("What is your cholesterol level?",
                           ("Normal", "Above normal", "Well above normal"))
gluc = st.selectbox("What is your glucose level?",
                           ("Normal", "Above normal", "Well above normal"))

gender = st.radio("What's your gender?", ["Men", "Women"])

smoke = st.radio("Do you smoke?", ["Yes", "No"], key="smoke")

alco = st.radio("Do you drink alcohol?", ["Yes", "No"], key="alco")

active = st.radio("Do you do physical activity?", ["Yes", "No"], key="active")


if st.button("Calculate"):

    predictor_numerical_test.loc[0,"age"] = relativedelta(datetime.datetime.now(), birth).years
    predictor_numerical_test.loc[0,"height"] = height
    predictor_numerical_test.loc[0,"weight"] = weigth
    predictor_numerical_test.loc[0,"ap_hi"] = ap_hi
    predictor_numerical_test.loc[0,"ap_lo"] = ap_lo
    predictor_nominal_test.loc[0,"gender"] = gender_id[gender]
    predictor_nominal_test.loc[0,"cholesterol"] = three_level_id[cholesterol]
    predictor_nominal_test.loc[0,"gluc"] = three_level_id[gluc]
    predictor_nominal_test.loc[0,"smoke"] = two_level_id[smoke]
    predictor_nominal_test.loc[0,"alco"] = two_level_id[alco]
    predictor_nominal_test.loc[0,"active"] = two_level_id[active]

    predictor_numerical_test = (predictor_numerical_test- mean)/std

    for column in predictor_numerical_test.columns:
        predictor_test.loc[0,column] = predictor_numerical_test.loc[0,column]

    for column in predictor_nominal_test.columns:
        predictor_test.loc[0,column+"_"+predictor_nominal_test.loc[0,column]] = 1

    st.metric(label="Cardiovascular Disease Risk",
            value=str(np.round(np.squeeze(model.predict(predictor_test)*100),2))+"%")

