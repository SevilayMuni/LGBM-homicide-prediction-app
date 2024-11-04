import streamlit as st
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd

st.title('🕵🏻 Homicide Solve Prediction App')
st.write('The project is dedicated to all murder victims and their families whose justice has not been served yet.')
st.info('The project aims to conduct data science research and demonstrate the importance of accurately accounting for unsolved homicides within communities.')
st.write('The model data source is Murder Accountability Project')

# Input features
with st.sidebar:
  st.header('Input Features')
  Agentype = st.selectbox('Agentype', ('local', 'state', 'federal'))
  Year = st.slider('Year', 1976, 2021)
  Month = st.slider('Month', 1, 12)
  Murder = st.selectbox('Murder', (1, 0))
  VicAge = st.slider('VicAge', 0, 99)
  VicSex = st.selectbox('VicSex', ('Male', 'Female', 'Unknown'))
  VicRace = st.selectbox('VicRace', ('unknown', 'white', 'indian', 'black', 'asian', 'islander'))
  Weapon = st.selectbox('Weapon', ('handgun', 'sharp object', 'other', 'explosives', 'blunt object', 'firearm', 'personal weapons - beating', 'strangulation', 'pushed - thrown', 'fire', 'drugs', 'drowning', 'asphyxiation', 'poison', 'not reported'))
  VicSex = st.selectbox('VicSex', ('Male', 'Female', 'Unknown'))



st.write('**model**')
pickled_model_app = pickle.load(open('model_app.pkl', 'rb'))
def process(dict):
    user_df = pd.DataFrame(dict)
    categ_col = user_df.select_dtypes(include = ['object']).columns.to_list()
    user_df[categ_col] = user_df[categ_col].astype('category')
    pred = round(pickled_model_app.predict(user_df).item())
    return pred

user_dict = {'Agentype': [Agentype], 'Year': [Year], 'Month': [Month], 
             'Murder': [Murder], 'VicAge': [VicAge], 'VicSex': [VicSex], 
             'VicRace':[VicRace], 'Weapon': [Weapon], 
             'Relationship': [Relationship], 'Circumstance':[Circumstance], 
             'VicCount': [VicCount], 'Region':[Region]}


