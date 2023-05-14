import streamlit as st
from PIL import Image

import pandas as pd

import numpy as np

import pickle

# import tensorflow.keras.backend as K
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import metrics
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

from functions_local import (get_data, make_spider, make_plots, predict_playstyle,
                       r_squared, predict_rank)


with open('et_model_5.pkl', 'rb') as picklefile:
    et_model = pickle.load(picklefile)

rank_model = keras.models.load_model('nn_model_tf_save.h5')

with open('ss.pkl', 'rb') as picklefile:
    ss_rank = pickle.load(picklefile)

with open('dict_ranks.pkl', 'rb') as picklefile:
    dict_ranks = pickle.load(picklefile)

rank_images = {}
rank_images['Bronze 1 Division 1'] = Image.open('ranks/b1.png')
rank_images['Bronze 1 Division 2'] = Image.open('ranks/b1.png')
rank_images['Bronze 1 Division 3'] = Image.open('ranks/b1.png')
rank_images['Bronze 1 Division 4'] = Image.open('ranks/b1.png')
rank_images['Bronze 2 Division 1'] = Image.open('ranks/b2.png')
rank_images['Bronze 2 Division 2'] = Image.open('ranks/b2.png')
rank_images['Bronze 2 Division 3'] = Image.open('ranks/b2.png')
rank_images['Bronze 2 Division 4'] = Image.open('ranks/b2.png')
rank_images['Bronze 3 Division 1'] = Image.open('ranks/b3.png')
rank_images['Bronze 3 Division 2'] = Image.open('ranks/b3.png')
rank_images['Bronze 3 Division 3'] = Image.open('ranks/b3.png')
rank_images['Bronze 3 Division 4'] = Image.open('ranks/b3.png')

rank_images['Silver 1 Division 1'] = Image.open('ranks/s1.png')
rank_images['Silver 1 Division 2'] = Image.open('ranks/s1.png')
rank_images['Silver 1 Division 3'] = Image.open('ranks/s1.png')
rank_images['Silver 1 Division 4'] = Image.open('ranks/s1.png')
rank_images['Silver 2 Division 1'] = Image.open('ranks/s2.png')
rank_images['Silver 2 Division 2'] = Image.open('ranks/s2.png')
rank_images['Silver 2 Division 3'] = Image.open('ranks/s2.png')
rank_images['Silver 2 Division 4'] = Image.open('ranks/s2.png')
rank_images['Silver 3 Division 1'] = Image.open('ranks/s3.png')
rank_images['Silver 3 Division 2'] = Image.open('ranks/s3.png')
rank_images['Silver 3 Division 3'] = Image.open('ranks/s3.png')
rank_images['Silver 3 Division 4'] = Image.open('ranks/s3.png')

rank_images['Gold 1 Division 1'] = Image.open('ranks/g1.png')
rank_images['Gold 1 Division 2'] = Image.open('ranks/g1.png')
rank_images['Gold 1 Division 3'] = Image.open('ranks/g1.png')
rank_images['Gold 1 Division 4'] = Image.open('ranks/g1.png')
rank_images['Gold 2 Division 1'] = Image.open('ranks/g2.png')
rank_images['Gold 2 Division 2'] = Image.open('ranks/g2.png')
rank_images['Gold 2 Division 3'] = Image.open('ranks/g2.png')
rank_images['Gold 2 Division 4'] = Image.open('ranks/g2.png')
rank_images['Gold 3 Division 1'] = Image.open('ranks/g3.png')
rank_images['Gold 3 Division 2'] = Image.open('ranks/g3.png')
rank_images['Gold 3 Division 3'] = Image.open('ranks/g3.png')
rank_images['Gold 3 Division 4'] = Image.open('ranks/g3.png')

rank_images['Platinum 1 Division 1'] = Image.open('ranks/p1.png')
rank_images['Platinum 1 Division 2'] = Image.open('ranks/p1.png')
rank_images['Platinum 1 Division 3'] = Image.open('ranks/p1.png')
rank_images['Platinum 1 Division 4'] = Image.open('ranks/p1.png')
rank_images['Platinum 2 Division 1'] = Image.open('ranks/p2.png')
rank_images['Platinum 2 Division 2'] = Image.open('ranks/p2.png')
rank_images['Platinum 2 Division 3'] = Image.open('ranks/p2.png')
rank_images['Platinum 2 Division 4'] = Image.open('ranks/p2.png')
rank_images['Platinum 3 Division 1'] = Image.open('ranks/p3.png')
rank_images['Platinum 3 Division 2'] = Image.open('ranks/p3.png')
rank_images['Platinum 3 Division 3'] = Image.open('ranks/p3.png')
rank_images['Platinum 3 Division 4'] = Image.open('ranks/p3.png')

rank_images['Diamond 1 Division 1'] = Image.open('ranks/d1.png')
rank_images['Diamond 1 Division 2'] = Image.open('ranks/d1.png')
rank_images['Diamond 1 Division 3'] = Image.open('ranks/d1.png')
rank_images['Diamond 1 Division 4'] = Image.open('ranks/d1.png')
rank_images['Diamond 2 Division 1'] = Image.open('ranks/d2.png')
rank_images['Diamond 2 Division 2'] = Image.open('ranks/d2.png')
rank_images['Diamond 2 Division 3'] = Image.open('ranks/d2.png')
rank_images['Diamond 2 Division 4'] = Image.open('ranks/d2.png')
rank_images['Diamond 3 Division 1'] = Image.open('ranks/d3.png')
rank_images['Diamond 3 Division 2'] = Image.open('ranks/d3.png')
rank_images['Diamond 3 Division 3'] = Image.open('ranks/d3.png')
rank_images['Diamond 3 Division 4'] = Image.open('ranks/d3.png')

rank_images['Champion 1 Division 1'] = Image.open('ranks/c1.png')
rank_images['Champion 1 Division 2'] = Image.open('ranks/c1.png')
rank_images['Champion 1 Division 3'] = Image.open('ranks/c1.png')
rank_images['Champion 1 Division 4'] = Image.open('ranks/c1.png')
rank_images['Champion 2 Division 1'] = Image.open('ranks/c2.png')
rank_images['Champion 2 Division 2'] = Image.open('ranks/c2.png')
rank_images['Champion 2 Division 3'] = Image.open('ranks/c2.png')
rank_images['Champion 2 Division 4'] = Image.open('ranks/c2.png')
rank_images['Champion 3 Division 1'] = Image.open('ranks/c3.png')
rank_images['Champion 3 Division 2'] = Image.open('ranks/c3.png')
rank_images['Champion 3 Division 3'] = Image.open('ranks/c3.png')
rank_images['Champion 3 Division 4'] = Image.open('ranks/c3.png')

rank_images['Grand Champion 1 Division 1'] = Image.open('ranks/gc1.png')
rank_images['Grand Champion 1 Division 2'] = Image.open('ranks/gc1.png')
rank_images['Grand Champion 1 Division 3'] = Image.open('ranks/gc1.png')
rank_images['Grand Champion 1 Division 4'] = Image.open('ranks/gc1.png')
rank_images['Grand Champion 2 Division 1'] = Image.open('ranks/gc2.png')
rank_images['Grand Champion 2 Division 2'] = Image.open('ranks/gc2.png')
rank_images['Grand Champion 2 Division 3'] = Image.open('ranks/gc2.png')
rank_images['Grand Champion 2 Division 4'] = Image.open('ranks/gc2.png')
rank_images['Grand Champion 3 Division 1'] = Image.open('ranks/gc3.png')
rank_images['Grand Champion 3 Division 2'] = Image.open('ranks/gc3.png')
rank_images['Grand Champion 3 Division 3'] = Image.open('ranks/gc3.png')
rank_images['Grand Champion 3 Division 4'] = Image.open('ranks/gc3.png')

rank_images['Supersonic Legend'] = Image.open('ranks/ssl.png')





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

        try:

            rank_preds = predict_rank(id_data, model = rank_model, ss = ss_rank, dict_ranks = dict_ranks)
        except KeyError:
            failed = True
        
        if failed:
            return st.write(f'''Problem loading the data - Check that the player name matches the name used in this game
                     https://ballchasing.com/replay/{user_input}''')

        # Display the output
        st.write(id_data)

        st.markdown('---')

        st.markdown('### Playstyle prediction')

        st.markdown('''
        - Your stats have been processed and used to produce playstyle metrics which are shown below:
            - Click the metrics below to see the factors that contribute to this metric
        ''')
        with st.expander('#### Speed :zap: :zap:'):
            st.markdown('''
            - average movement speed
            - boost consumed per minute
            - powerslide count
            - the ratio of time spent in high air : low air and ground
            - the ratio of time spent at max speed : 'slow' speed
            ''')
                
        with st.expander('#### Boost Efficiency :brain:'):
            st.markdown('''
            - #### Capped at 2 * Speed metric
            - average amount of boost
            - the ratio of small boost pads collected : big boost pads
            - average speed and boost usage
            - amount of boost overfill (collecting big boosts while already on high boost amount will lead to high overfill values)
            - amount of boost used while already max speed (supersonic)
            - time spent at zero boost
            - percent of time spent at over 50 boost vs percent of time spend below 50 boost
            ''')

        with st.expander('#### Aggression :fire:'):
            st.markdown('''
            - stealing boost
            - using boost while supersonic
            - playing far from teammates
            - inflicting demolitions
            - ratio of time spent in the opposition third of the pitch : defensive third of the pitch
            - percentage of time spent in front of the ball
            - percentage of time as the furthest forward team member on the pitch
            ''')

        with st.expander('#### Team Cohesion :handshake:'):
            st.markdown('''
            - #### Capped at 2 * Speed metric
            - small boost pad : big boost pad collection ratio
            - percentage of time spent most back
            - distance to teammates
            ''')
            
        with st.expander('#### Game Involvement :trophy:'):
            st.markdown('''
            - total score
            - stealing boost from opposition
            - percentage of time spent off the ground
            - percentage of time spent closest to the ball
            ''')
        st.pyplot(radar_plots)
        st.write(df_stats.round(2).rename(columns=lambda x: ' '.join(x.split())
        ) )
        pros_ps = st.image('2-cleaning-EDA/code/pros-playstyle.png')
        st.markdown('---')
        st.markdown('### Probabilities')
        col1, col2, col3 = st.columns(3)
        dict_players = {
            0 : 'Monkey Moon',
            1 : 'Oski',
            2 : 'Vatira'
        }
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
        if player_pred == 'M0nkey M00n':
            st.markdown('## :fr:')
            st.markdown('''
                1. Monkey Moon's playstyle - while slower than most professional players - involes being consistantly available to receive a pass or intercept an opposition
                2. Their ability to play at reasonably high speed whilst maintaining enough boost to contribute at all times makes them a very valuable teammate
                3. Monkey Moon has arguably the strongest decision making - only using the minimum resources required to make a positive play
            ''')
        # if player_pred.iloc[0].idxmax() == 1:
        if player_pred == 'Oski':
            st.markdown(''' 
                :white_circle: \n
                :red_circle:''')
            st.markdown('''
                1. Oski plays as aggressively as possible often leaving teammates in unfavourable scenarios. Oski's hyper fast playstyle however has generated such a number of opportunities for the team that this tradeoff has produced great results for their team
                2. Characteristics of this playstyle are: 
                - high boost consumption
                - high average speed
                - high percentage of time spent furthest forward on the team
            ''')
        # if player_pred.iloc[0].idxmax() == 2:
        if player_pred == 'Vatira':
            st.markdown('## :fr:')
            st.markdown('''
                1. Vatira has a very well rounded playstyle, and is able to play to the strengths of any teammates.
                2. They typically play more defensively than other teammates, however often deliver the majority of goals nonetheless.
                3. Vatira uses massive technical ability to capitalise on space generated by more aggressive teammates, for this reason they tend to spend a comparitively high amount of time on high boost amounts, and maintain these levels through strong boost efficiency.

            '''
            )
        st.markdown('---')
        rank_str = ''.join(rank_preds)
        
        col1_1, col2_1 = st.columns(2)
        with col1_1:
            st.markdown('# Rank Predictions')

            st.markdown(f"## {rank_str}")

        with col2_1:
            st.image(rank_images[rank_str])

        st.markdown('---')

         


analyse_game(user_input, player_name)

if show_samples == True and not submit:
    st.markdown('---')
    st.markdown('## Pro Replays')
    st.markdown('---')

    st.write('''**M0nkey M00n** replay 1:

    0c4c75e7-55cb-4a6a-b531-f32959b00acb           
    ''')
    st.markdown('---')

    st.write('''**Oski** replay 1:

    fcce2e38-2ecc-4bf3-838a-41029c0b3842        
    ''')
    st.markdown('---')

    st.write('''**Vatira** replay 1:

    29349d8a-d32f-4d6a-9624-46514f141ca2            
    ''')
    st.markdown('---')

    st.write('''**Marc_by_8** replay 1:

    fcce2e38-2ecc-4bf3-838a-41029c0b3842         
    ''') 
    st.markdown('---')

    st.markdown('## Mid-High Rank Replays')
    st.markdown('---')


    st.write('''**boay00** replay 1:

    305e79bd-9b27-40c6-b0b6-66a73063f3dd
    
    ''')
    st.write('')

    # image = Image.open('champ-2.jpeg')
    # st.image(image)

    st.write('''**boay00** replay 2:

    0d3d3bc6-f09b-4d77-9047-8afa863334ce
    ''')
    st.write('')

    st.write('''**boay00** replay 3:

    36778dbd-048b-46e3-aef1-94a3895d2543
    ''')

    st.markdown('---')

    st.write('''**Cho Shmo** replay 1:

    b7e5be3f-50ff-4ff5-ac29-0a020d0dc327        
    ''')
    
    st.markdown('---')

    st.write('''**eden** replay 1:

    3e502845-e51d-43bc-895a-5cfd61e52460        
    ''')
    st.write('')
    st.write('''**eden** replay 2:

    305e79bd-9b27-40c6-b0b6-66a73063f3dd        
    ''')
    
    st.markdown('---')
    st.markdown('## Low-Mid Rank Replays')
    st.markdown('---')

    st.write('''**bdpie1** replay 1:

    0d3d3bc6-f09b-4d77-9047-8afa863334ce
    ''')
    st.markdown('---')

    st.write('''**RiddlingTuna546** replay 1:

    73b8bf4e-aeee-48cc-ad89-9902dadc0537          
    ''')    
    
    st.markdown('---')

    st.write('''**KitchenView** replay 1:

    65d388c7-ffc4-4f96-85d0-f96250a6638e         
    ''') 
    
