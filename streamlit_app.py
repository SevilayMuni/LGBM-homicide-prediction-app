import streamlit as st
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib

pickled_model_app = joblib.load('./model_app.pkl')

st.title('🕵🏻 Homicide Solve Prediction App')
st.write('The project is dedicated to all murder victims and their families whose justice has not been served yet.')
st.info('The project aims to conduct data science research and demonstrate the importance of accurately accounting for unsolved homicides within communities. The model data source is Murder Accountability Project')

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
  Relationship = st.selectbox('Relationship', ('unknown', 'girlfriend', 'stranger', 'other', 'brother', 'acquaintance', 'father', 'second-degree relative', 'friend', 'stepdaughter', 'husband', 'wife', 'ex-husband', 'boyfriend', 'stepfather', 'son', 'ex-wife', 'sister', 'mother', 'stepson', 'neighbor', 'daughter', 'employer', 'stepmother', 'in-law', 'employee'))
  Circumstance = st.selectbox('Circumstance', ('other argument', 'other', 'all-suspected-felony', 'brawl', 'lovers triangle', 'undetermined', 'robbery', 'money - property argument', 'felon killed by private citizen', 'negligent gun handling', 'motor vehicle theft', 'prostitution', 'burglary', 'narcotics', 'felon killed by police', 'arson', 'larceny', 'sex offense', 'manslaughter by negligence', 'child killed by babysitter', 'juvenile gang killings', 'gangland killings', 'gambling', 'sniper attack', 'institutional killings'))
  VicCount = st.slider('VicCount', 1, 40)
  Region = st.selectbox('Region', ('west', 'south', 'northeast', 'midwest'))

# Prepare Input Data
user_dict = {'Agentype': [Agentype], 'Year': [Year], 'Month': [Month], 
             'Murder': [Murder], 'VicAge': [VicAge], 'VicSex': [VicSex], 
             'VicRace':[VicRace], 'Weapon': [Weapon], 
             'Relationship': [Relationship], 'Circumstance':[Circumstance], 
             'VicCount': [VicCount], 'Region':[Region]}

def process(user_dict):
    user_df = pd.DataFrame(user_dict)
    categ_col = user_df.select_dtypes(include = ['object']).columns.to_list()
    user_df[categ_col] = user_df[categ_col].astype('category')
    pred = round(pickled_model_app.predict(user_df).item())
    return pred

prediction = process(user_dict)
predicted_class = {0: 'UNSOLVED', 1: 'SOLVED'}[prediction]

# Display result
st.write(f"Probable Result: {predicted_class}")

# predicted_class = {0: 'unsolved', 1: 'solved'}

