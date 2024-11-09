import streamlit as st
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib

pickled_model_app = joblib.load('./model_app.pkl')

tab1, tab2, tab3 = st.tabs(["Prediction", "Info", "Charts"])

tab1.header('🕵🏻 Homicide Solve Prediction App')
tab1.subheader('The project is dedicated to all murder victims and their families whose justice has not been served yet.')
tab1.success('The project aims to conduct data science research and demonstrate the importance of accurately accounting for unsolved homicides within communities.')
tab1.subheader('', divider = 'rainbow')

with tab2.expander("Infographics"):
    st.markdown('''Info on dataset''')
    st.link_button("Data Source: Murder Accountability Project", "https://www.murderdata.org/")


# Input features
with st.sidebar:
  st.header('INPUT FEATURES')
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
predicted_class = {0: 'UNSOLVED!', 1: 'SOLVED!'}[prediction]

# Display result
tab1.info(f"Predicted Result: {predicted_class}")
tab1.markdown(''':rainbow[End-to-end project is done by] and :blue-background[Sevilay Munire Girgin]''')

with tab2.expander("Feature Importance Plot"):
    st.markdown(''':violet[The chart above shows lightGBM feature importance plot based on gain.  
        It states what features heavily impacted the model's decision.]''')
    st.image("./images2/Gain-Feature-Importance-Plot.png")

with tab2.expander("Feature Importance Plot"):
    st.markdown(''':violet[The chart above shows lightGBM feature importance plot based on gain.  
        It states what features heavily impacted the model's decision.]''')
    st.image("./images2/Gain-Feature-Importance-Plot.png")

with tab3.popover("Charts on Victim Data"):
    st.markdown("Hello World 👋")
    name = st.text_input("What's your name?")

with tab3.popover("Charts on Offender Data"):
    st.markdown("Hello World 👋")
    name = st.text_input("What's your name?")

container1 = tab3.container(border = True)
container1.image("./images2/case-year-histogram.png")
container2 = tab3.container(border = True)
container2.image("./images2/victim-age-VS-offender-race.png")


