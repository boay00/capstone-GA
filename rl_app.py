import streamlit as st
from PIL import Image

import requests
import json
import pandas as pd
import time
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

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from functions import (get_data, make_spider, make_plots, predict_playstyle,
                       r_squared, predict_rank)

st.title('Capstone GA')
st.set_option('deprecation.showPyplotGlobalUse', False)


player_name = st.text_input('Enter your gamertag here', value = '', key = 'input')
user_input = st.text_input('Enter your replay ID here', value = '', key = 'input2')

submit = st.button("Submit")

if not user_input and player_name:
    st.error("If you want to see a demo - select from a sample ID!")
    show_samples = True
show_samples = st.checkbox("Show Sample Replays IDs")

if show_samples == True:
    st.write('''boay00 replay 1:

    305e79bd-9b27-40c6-b0b6-66a73063f3dd
    
    ''')
    st.write('')

    # image = Image.open('champ-2.jpeg')
    # st.image(image)

    st.write('''boay00 replay 2:

    0d3d3bc6-f09b-4d77-9047-8afa863334ce
    ''')
    st.write('')

    st.write('''boay00 replay 3:

    36778dbd-048b-46e3-aef1-94a3895d2543
    ''')

    st.markdown('---')

    st.write('''bdpie1 replay 1:

    0d3d3bc6-f09b-4d77-9047-8afa863334ce
    ''')
    st.markdown('---')

    st.write('''Cho Shmo replay 1:

    b7e5be3f-50ff-4ff5-ac29-0a020d0dc327        
    ''')
    
    st.markdown('---')

    st.write('''eden replay 1:

    3e502845-e51d-43bc-895a-5cfd61e52460        
    ''')


if submit and player_name and user_input:
    show_samples = False
    # Do something with the inputs
    id_data = get_data(user_input, player_name)

 

    # Display the output
    st.write(id_data)

    st.markdown('---')

    st.markdown('### Playstyle prediction')

    radar_plots = make_plots(id_data, player_name)
    st.pyplot(radar_plots)
    ps_preds = predict_playstyle(id_data)
    st.write(ps_preds)
    st.markdown('---')
    st.write('Rank Predictions')
    rank_preds = predict_rank(id_data)
    st.write(rank_preds)






