import numpy as np
import pandas as pd
import kagglehub
import os
import altair as alt
from scipy.stats import chi2
import streamlit as st

def MahalanobisDist(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            vars_mean = []
            for i in range(data.shape[0]):
                vars_mean.append(list(data.mean(axis=0)))
            diff = data - vars_mean
            md = []
            for i in range(len(diff)):
                md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))

            if verbose:
                print("Covariance Matrix:\n {}\n".format(covariance_matrix))
                print("Inverse of Covariance Matrix:\n {}\n".format(inv_covariance_matrix))
                print("Variables Mean Vector:\n {}\n".format(vars_mean))
                print("Variables - Variables Mean Vector:\n {}\n".format(diff))
                print("Mahalanobis Distance:\n {}\n".format(md))
            return md
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

@st.cache_data
def get_dataset():
    path = kagglehub.dataset_download("sulianova/cardiovascular-disease-dataset")
    dataset_full = pd.read_csv(os.path.join(path, "cardio_train.csv"), delimiter=";") 
    dataset = dataset_full.sample(5000)  
    dataset.loc[:,'age'] = dataset['age']/365
    predictor_numerical =  dataset[["age", "height", "weight", "ap_hi", "ap_lo"]].astype(float)
    crit = chi2.ppf(1-0.25, df=len(predictor_numerical.columns)-1)
    mahalanobis = MahalanobisDist(predictor_numerical.to_numpy())
    dataset_new = dataset[mahalanobis < crit]

    return dataset_new

dataset = get_dataset()

numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
nominal_columns = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]

st.title("Cardiovascular Disease Visualization")
st.write("This is description.")

tab1, tab2, tab3 = st.tabs(["Donut Chart", "Bar Chart", "Scatter Plot"])

with tab1:

    c1_color = st.selectbox("Select nominal variable:", nominal_columns, key="c1_color")

    c1 = (alt.Chart(dataset).mark_arc(innerRadius=50).encode(
    theta="count()",
    color=alt.Color(c1_color, type="nominal"),
    tooltip=['count()']))

    st.altair_chart(c1, use_container_width=True)

with tab2:

    c2_x = st.selectbox("Select numerical variable:", numerical_columns)
    c2_color = st.selectbox("Select nominal variable:", nominal_columns, key="c2_color")

    c2 = (alt.Chart(dataset).mark_bar().encode(
    x=alt.X(c2_x, bin=True, type="quantitative"),
    y='count()',
    color=alt.Color(c2_color, type="nominal")))

    st.altair_chart(c2, use_container_width=True)

with tab3:

    c3_x = st.selectbox("Select numerical variable for axis-x:", numerical_columns)
    c3_y = st.selectbox("Select numerical variable for axis-y:", numerical_columns)
    c3_color = st.selectbox("Select nominal variable:", nominal_columns, key="c3_color")

    c3 = (alt.Chart(dataset).mark_point().encode(
    x=alt.X(c3_x, type="quantitative"),
    y=alt.Y(c3_y, type="quantitative"),
    color=alt.Color(c3_color, type="nominal")).interactive())

    st.altair_chart(c3, use_container_width=True)