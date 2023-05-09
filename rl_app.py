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

from functions import get_data, make_spider, make_plots

st.title('Capstone GA')
st.set_option('deprecation.showPyplotGlobalUse', False)


player_name = st.text_input('Enter your gamertag here', value = '', key = 'input')
user_input = st.text_input('Enter your replay ID here', value = '', key = 'input2')

if st.button("Submit") and player_name and user_input:
    # Do something with the inputs
    id_data = get_data(user_input, player_name)

    show_samples = False

    # Display the output
    st.write(id_data)
    speed = make_plots(id_data, player_name)
    st.write(speed)


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

    d0971503-bba7-41a4-bfc8-129a3f479ce3
    ''')



