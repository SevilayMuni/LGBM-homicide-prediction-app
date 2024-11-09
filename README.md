# 🕵🏻 Justice Forecast: Solvability Analysis for Homicide Cases 👩🏻‍💻

## The project is dedicated to all murder victims and their families whose justice has not been served yet.

The project aims to conduct data science research and demonstrate the importance of accurately accounting for unsolved homicides within communities. 

The model data source is [Murder Accountability Project](https://www.murderdata.org/)

## Check Out My Application

[![My Prediction App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sevilaygirgin-app-homicide-prediction.streamlit.app/)


## Model Evaluation
*Feature importance plot based on **gain**. It quantifies the model’s accuracy improvement achieved by using specific features for splitting.*
[<img src="https://github.com/SevilayMuni/LGBM-homicide-prediction-app/blob/master/images2/Gain-Feature-Importance-Plot.png" width="500"/>](https://github.com/SevilayMuni/LGBM-homicide-prediction-app/blob/master/images2/Gain-Feature-Importance-Plot.png)

*Feature importance plot based on **split**. This measures the number of times a feature is used to split the data across all trees in the model.*
[<img src="https://github.com/SevilayMuni/LGBM-homicide-prediction-app/blob/master/images2/Split-Feature-Importance-Plot.png" width="500"/>](https://github.com/SevilayMuni/LGBM-homicide-prediction-app/blob/master/images2/Split-Feature-Importance-Plot.png)

All in all, **important features for homicide solvability**: 

    Relationship between the victim and offender
    The circumstance (or theory) of the crime
    Year of the homicide
    Victim's Age

## Variable Description Table

Feature | Description |
-----|-----|
Agentype| Type of the law enforcement agency
Year| Year of homicide (or victim’s body was recovered) 
Month| The month of homicide (or victim’s body was recovered)
Murder| 1: Murder & 0: Negligent Manslaughter
VicAge| Victim’s age
VicSex| Victim’s sex (“Unknown” gender: incomplete remains were recovered)
VicRace| Victim’s race
Weapon| Weapon used in the crime
Relationship| The relationship between victim and offender
Circumstance| The circumstances (or theory) of the crime
VicCount| Victim number in the crime
Region| USA region in which the homicide was reported

## Contact 📩
For any questions or inquiries, feel free to reach out:
- [**My Website:**](https://sevilaymuni.github.io/Girgin/)
- **LinkedIn:** [Sevilay Munire Girgin](www.linkedin.com/in/sevilay-munire-girgin-8902a7159)

Thank you for visiting my project repository. Happy and accurate classification! 💕

| One's destination is never a place but rather a new way of seeing things. - Henry Miller |
-----|

<p align="center">
// ***Copyright (c) 2024 Sevilay Münire Girgin***
</p>
