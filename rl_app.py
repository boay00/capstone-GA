import streamlit as st
from PIL import Image

import requests
import json
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
import scipy as scp
from math import pi


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import os
import pickle

# import tensorflow.keras.backend as K
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import metrics
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping

from functions import (get_data, make_spider, make_plots, predict_playstyle,
                       r_squared, predict_rank)


with open('et_model_4.pkl', 'rb') as picklefile:
    et_model = pickle.load(picklefile)

with open('gbrt_5.pkl', 'rb') as picklefile:
    rank_model = pickle.load(picklefile)

with open('dict_ranks.pkl', 'rb') as picklefile:
    dict_ranks = pickle.load(picklefile)

st.title('Prediction HUB')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('''---
#### If you have played Rocket League, navigate to [this link](https://ballchasing.com), and enter your gamertag in the right search bar
- Please filter for any of these 3v3 game modes:
    - 'Ranked Solo Standard'
    - 'Ranked Standard'
    - 'Unranked Standard'
- (Techincally any 3v3 game mode will work, so tournaments and rumble, dropshot, snowday would still work! My models have never seen data from extra modes - so try for fun! :grin:)
- Only players on PC can upload replays to ballchasing, although console players can still find games other people have uploaded with them also playing in!
- NB. Players at ranks of diamond or higher are fare more likely to find replays - since lower ranked players rarely upload replays to ballchasing.


''')

player_name = st.text_input('Enter your gamertag here', value = '', key = 'input')
user_input = st.text_input('Enter your replay ID here', value = '', key = 'input2')

submit = st.button("Submit")
show_samples = st.checkbox("Show Sample Replays IDs")

if not user_input:
    error = st.error("If you want to see a demo - select from a sample ID!")
    

def analyse_game(user_input, player_name):
    failed = False

    if submit and player_name and user_input:
        # Do something with the inputs
        try:

            id_data = get_data(user_input, player_name)
        except KeyError:
            failed = True
        
        if failed:
            return st.write(f'''Problem loading the data - Check that the player name matches the name used in this game
                     https://ballchasing.com/replay/{user_input}''')
            
        try:

            radar_plots, df_stats = make_plots(id_data, player_name)
        except KeyError:
            failed = True
        
        if failed:
            return st.write(f'''Problem loading the data - Check that the player name matches the name used in this game
                     https://ballchasing.com/replay/{user_input}''')
        try:

            ps_preds, player_pred = predict_playstyle(id_data, model = et_model)
        except KeyError:
            failed = True
        
        if failed:
            return st.write(f'''Problem loading the data - Check that the player name matches the name used in this game
                     https://ballchasing.com/replay/{user_input}''')

        # try:

        #     rank_preds = predict_rank(id_data, model = rank_model, dict_ranks = dict_ranks)
        # except KeyError:
        #     failed = True
        
        # if failed:
        #     return st.write(f'''Problem loading the data - Check that the player name matches the name used in this game
        #              https://ballchasing.com/replay/{user_input}''')

        # Display the output
        st.write(id_data)

        st.markdown('---')

        st.markdown('### Playstyle prediction')

        
        st.pyplot(radar_plots)
        st.write(df_stats.round(2).rename(columns=lambda x: ' '.join(x.split())
        ) )
        pros_ps = st.image('2-cleaning-EDA/code/pros-playstyle.png')
        st.markdown('---')
        st.markdown('### Probabilities')
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown(f'''<p style="color:turquoise">
                ------------ MonkeyMoon : {'{:.2f}'.format(ps_preds.values[0][0].round(2))} ----''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''<p style="color:#8bc34a">
                          ----------------- Oski : {'{:.2f}'.format(ps_preds.values[0][1].round(2))} ------------''', unsafe_allow_html=True)
        with col3:
            st.markdown(f'''<p style="color:#FFD700">
                          -------------- Vatira : {'{:.2f}'.format(ps_preds.values[0][2].round(2))} ------------''', unsafe_allow_html=True)
        st.write(f'''Your playstyle assciates the most with:''')
        st.markdown(f"## {ps_preds.columns[ps_preds.values.argmax()]} -> {max('{:.2f}'.format(ps_preds.values[0][0].round(2)), '{:.2f}'.format(ps_preds.values[0][1].round(2)), '{:.2f}'.format(ps_preds.values[0][2].round(2)))}")
        # st.markdown(f'## {dict_players[player_pred.iloc[0].idxmax()]} -> {"{:.2f}".format(ps_preds.values[0][player_pred.iloc[0].idxmax()].round(2))}')
        # if player_pred.iloc[0].idxmax() == 0:

        # st.markdown('---')
        # rank_str = ''.join(rank_preds)
        
 
    
        # st.markdown('# Rank Predictions')

        # st.markdown(f"## {rank_str}")

 

         


analyse_game(user_input, player_name)

if show_samples == True and not submit:
    st.markdown('---')
    st.markdown('## Pro Replays')
    st.markdown('---')

    st.write('''**M0nkey M00n** replay 1:

    0c4c75e7-55cb-4a6a-b531-f32959b00acb           
    ''')

    
