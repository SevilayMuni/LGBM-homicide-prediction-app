import streamlit as st
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd

st.title('🕵🏻 Homicide Solve Prediction App')
st.write('The project is dedicated to all murder victims and their families whose justice has not been served yet.')
st.info('The project aims to conduct data science research and demonstrate the importance of accurately accounting for unsolved homicides within communities.')
st.write('The model data source is Murder Accountability Project')

st.write('**model**')
pickled_model_app = pickle.load(open('model_app.pkl', 'rb'))
def process(dict):
    user_df = pd.DataFrame(dict)
    categ_col = user_df.select_dtypes(include = ['object']).columns.to_list()
    user_df[categ_col] = user_df[categ_col].astype('category')
    pred = round(pickled_model_app.predict(user_df).item())
    return pred


