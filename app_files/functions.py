import requests
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns

import pickle
from math import pi 

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import metrics
# import tensorflow.keras.backend as K

    

def get_data(replay_id, player_name):
    res = requests.get(f'https://ballchasing.com/api/replays/{replay_id}',
                  headers={
                      'Authorization': 'mPuO0QZqG0wXdB9gDZHp7KKI09KplWLsYXkJQzJ5'})

    data = res.json()
    
    player_data = data['orange']['players'][0]['stats']
    player_data = dict(player_data, **player_data['core'])
    player_data = dict(player_data, **player_data['boost'])
    player_data = dict(player_data, **player_data['movement'])
    player_data = dict(player_data, **player_data['positioning'])
    player_data = dict(player_data, **player_data['demo'])
    del player_data['core']
    del player_data['boost']
    del player_data['movement']
    del player_data['positioning']
    del player_data['demo']

#     player_data['title'] = data['title']
    player_data['player_name'] = player_name
    
    df = pd.DataFrame(columns = player_data.keys())
    
    try:
        for j in range(0, min([len(data['orange']['players']), len(data['blue']['players'])])):

            if player_name != None:
                if data['orange']['players'][j]['name'][:4].lower() == player_name[:4].lower():
                    player_data = data['orange']['players'][j]['stats']

                    player_data = dict(player_data, **player_data['core'])
                    player_data = dict(player_data, **player_data['boost'])
                    player_data = dict(player_data, **player_data['movement'])
                    player_data = dict(player_data, **player_data['positioning'])
                    player_data = dict(player_data, **player_data['demo'])
                    del player_data['core']
                    del player_data['boost']
                    del player_data['movement']
                    del player_data['positioning']
                    del player_data['demo']

                    player_data['title'] = data['title']
                    player_data['player_name'] = player_name
                    df.loc[player_data['title'], :] = player_data

                elif data['blue']['players'][j]['name'][:4].lower() == player_name[:4].lower():

                    player_data = data['blue']['players'][j]['stats']

                    player_data = dict(player_data, **player_data['core'])
                    player_data = dict(player_data, **player_data['boost'])
                    player_data = dict(player_data, **player_data['movement'])
                    player_data = dict(player_data, **player_data['positioning'])
                    player_data = dict(player_data, **player_data['demo'])
                    del player_data['core']
                    del player_data['boost']
                    del player_data['movement']
                    del player_data['positioning']
                    del player_data['demo']

                    player_data['title'] = data['title']
                    player_data['player_name'] = player_name

                    df.loc[player_data['title'], :] = player_data

    except KeyError:
        errors += 1
        print(f'KeyError in match {i}, errors = {errors}')
            

    return df


def make_spider(df, row, title, color):
    '''
    This function takes in a dataframe of player stats, and produces len(df) number of radar plots highlighting the playstyle of that player
    The y limit in floored a the minumum stat value - 1 and scaled according to the pro max metrics pickled above
    '''
    
    categories=list(df)[1:]
    N = len(categories)
    
    # set the angles of the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(1,len(df),row+1, polar=True)

    # first axis on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # one axe per variable and add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # ylabels set ylim to lowest int(value) - 1
    ax.set_rlabel_position(0)
    plt.yticks([1,2,3,4,5,6,7,8,9], ['1',"2",'3',"4",'5',"6",'7', "8",'9'], color="grey", size=7)
    lowest = df.select_dtypes(include = 'number').values.min()
    plt.ylim(min(5,max(0, (int(lowest) - 1))),10)

    values=df.loc[row].drop('player').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    plt.title(title, size=11, color=color, y=1.1)


with open('max_metrics.pickle', 'rb') as picklefile:
    max_speed, max_boost_eff, max_agg, max_team_cohesion, max_game_inv = pickle.load(picklefile)
    
def make_plots(df_original, player_name):
    
    '''
    This function performs the full process of cleaning, metric calculation and plot generation following data collection.
    It generates the mean for each stat, splits these into required sub dataframes, then runs the stats through the algorithm created
    which then produces the general playstyle stats
    '''
    
    df = df_original.copy()
    
    # means calculation and cleaning
    means = df.mean(axis= 0)
    df.index = range(len(df.index))
    df = df.drop(columns = ['player_name'])
    df = df.append(means, ignore_index = True)
    df = df.loc[[len(df) - 1], :]
    df = df.rename(index={len(df): player_name})

    # some games save this stat, others do not, so just drop all instances
    try:
        df.drop(columns = 'goals_against_while_last_defender')
    except KeyError:
        pass

    to_drop = [
        'shots_against',
        'goals_against',
        'shooting_percentage',
        'bpm',
        'amount_stolen_big',
        'amount_stolen_small',
        'count_collected_big',
        'count_collected_small',
        'count_stolen_small',
        'count_stolen_big',
        'amount_overfill_stolen',
        'time_zero_boost',
        'time_full_boost',
        'time_boost_0_25',
        'time_boost_25_50',
        'time_boost_50_75',
        'time_boost_75_100',
        'avg_speed',
        'total_distance',
        'time_supersonic_speed',
        'time_boost_speed',
        'time_slow_speed',
        'time_ground',
        'time_low_air',
        'time_high_air',
        'time_powerslide',
        'time_defensive_third',
        'time_neutral_third',
        'time_offensive_third',
        'time_defensive_half',
        'time_offensive_half',
        'time_behind_ball',
        'time_infront_ball',
        'time_most_back',
        'time_most_forward',
        'time_closest_to_ball',
        'time_farthest_from_ball'
    ]
    df.drop(columns = to_drop, inplace = True)

    # assorting stats into their relevant metric categories
    speed = ['count_powerslide', 
             'percent_supersonic_speed', 
             'avg_speed_percentage', 
             'percent_slow_speed',
             'percent_high_air',
             'percent_low_air',
             'percent_ground',
             'bcpm'
            ]

    boost_efficiency = ['bcpm',
                        'avg_amount',
                        'amount_collected',
                        'amount_collected_big',
                        'amount_collected_small',
                        'amount_overfill',
                        'amount_used_while_supersonic',
                        'percent_zero_boost',
                        'percent_full_boost',
                        'percent_boost_0_25',
                        'percent_boost_25_50',
                        'percent_boost_50_75',
                        'percent_boost_75_100',
                        'avg_powerslide_duration',
                        'avg_speed_percentage',
                        'percent_boost_speed',
                        'percent_ground',
                        'percent_low_air',
                        'percent_high_air'
    ]

    aggression = ['amount_stolen',
                  'amount_used_while_supersonic',
                  'avg_distance_to_mates',
                  'inflicted',
                  'percent_defensive_third',
                  'percent_offensive_third',
                  'percent_infront_ball',
                  'percent_most_back',
                  'percent_most_forward'
    ]

    team_cohesion = ['goals',
                     'assists',
                     'amount_collected_big',
                     'amount_collected_small',
                     'avg_distance_to_ball_possession',
                     'avg_distance_to_ball_no_possession',
                     'avg_distance_to_mates'
    ]

    game_involvement = ['score',
                        'amount_collected',
                        'amount_stolen',
                        'percent_low_air',
                        'percent_ground',
                        'avg_distance_to_ball_possession',
                        'avg_distance_to_ball_no_possession',
                        'percent_closest_to_ball',
                        'percent_farthest_from_ball',
                        'inflicted'
    ]

    # create dataframes for each of the 5 metrics
    speed_df = df.loc[:, speed]
    boost_efficiency_df = df.loc[:, boost_efficiency]
    aggression_df = df.loc[:, aggression]
    team_cohesion_df = df.loc[:, team_cohesion]
    game_involvement_df = df.loc[:, game_involvement]



    # speed stat calculation
    # each addition was weighted to roughly the same weight using trial and error
    speed_df['speed'] = (speed_df.count_powerslide / 10
                        ) + (speed_df.avg_speed_percentage / 10
                        ) + (speed_df.percent_high_air
                        ) + ((speed_df.percent_low_air / speed_df.percent_ground) * 9
                        ) + (speed_df.bcpm / 66) + ((speed_df.percent_supersonic_speed/speed_df.percent_slow_speed
                        ) * 15)
    # speed stat scaled using pro max metric value
    speed_df['speed'] = speed_df['speed'] / max_speed * 9.5
    
    # process repeated for each of the 5 metrics
    boost_efficiency_df['boost_efficiency'] = boost_efficiency_df.avg_amount * (((boost_efficiency_df.amount_collected_small / boost_efficiency_df.amount_collected
                                              ) * 25) + ((1/boost_efficiency_df.avg_speed_percentage) * (boost_efficiency_df.bcpm) 
                                              ) * 1.2 + (1 / (boost_efficiency_df.amount_overfill / boost_efficiency_df.amount_collected)
                                              ) + (0.8 / (boost_efficiency_df.amount_used_while_supersonic / boost_efficiency_df.amount_collected)
                                              ) + (75 / boost_efficiency_df.percent_zero_boost
                                              ) + (1 - (abs(boost_efficiency_df.avg_powerslide_duration - 0.1
                                              )) * 100) + (((boost_efficiency_df.percent_boost_50_75
                                                    + boost_efficiency_df.percent_boost_75_100)
                                                   / (boost_efficiency_df.percent_boost_0_25
                                                    + boost_efficiency_df.percent_boost_25_50)
                                                  ) * 10)
    ) / 250





    boost_efficiency_df['boost_efficiency'] = (boost_efficiency_df.boost_efficiency / max_boost_eff
                                                                                        ) * 9.5
    aggression_df['aggression'] = (aggression_df.amount_stolen / 100
                                  ) + ((aggression_df.amount_used_while_supersonic / 100
                                  ) * 1.66) + (aggression_df.avg_distance_to_mates / 540
                                  ) + (aggression_df.inflicted * 6
                                  ) + ((aggression_df.percent_offensive_third/aggression_df.percent_defensive_third
                                  ) * 12) + ((aggression_df.percent_infront_ball / 10
                                  ) * 2.2) + ((aggression_df.percent_most_forward/aggression_df.percent_most_back
                                  ) * 6)


    aggression_df['aggression'] = (aggression_df.aggression / max_agg) * 9.5


    team_cohesion_df['team_cohesion'] = (((team_cohesion_df.amount_collected_small / team_cohesion_df.amount_collected_big
                                        ) * (boost_efficiency_df.amount_collected) 
                                        ) / 150) + (aggression_df.percent_most_back/team_cohesion_df.avg_distance_to_mates) * 800


    team_cohesion_df['team_cohesion'] = (team_cohesion_df.team_cohesion / max_team_cohesion) * 9.5

    game_involvement_df['game_involvement'] = (game_involvement_df.score / 60
                                                ) + ((game_involvement_df.amount_stolen / game_involvement_df.amount_collected
                                                ) * 25
                                                ) + ((100 - game_involvement_df.percent_ground
                                                ) / 7
                                                ) + (game_involvement_df.percent_closest_to_ball / 5
                                                # ) + ((game_involvement_df.inflicted
                                                # ) * 7
                                                    ) + (game_involvement_df.score / 70)


    game_involvement_df['game_involvement'] = (game_involvement_df.game_involvement / max_game_inv) * 9.5
    ## radar plot code taken from github example and adapted for personal needs

    # new dataframe created for the metrics
    # Set data
    df_2 = pd.DataFrame({
    'player' : [player_name],
    """Speed""": min(10,speed_df.loc[player_name, 'speed']
    ),
    """  Boost 
            Efficiency""":     min(10,min(speed_df.loc[player_name, 'speed'] * 2, boost_efficiency_df.loc[player_name, 'boost_efficiency'])
                                  
    ),
    """  Aggression""": min(10,aggression_df.loc[player_name, 'aggression']

    ),
    """Team Cohesion       """: min(10, min(speed_df.loc[player_name, 'speed'] * 2, team_cohesion_df.loc[player_name, 'team_cohesion']
    ))
    ,
    """   Game    
    Involvement               """: min(10,game_involvement_df.loc[player_name, 'game_involvement']

    )
    })


    my_dpi=96
    plt.figure(figsize=(600/my_dpi, 600/my_dpi), dpi=my_dpi)

    # color palette
    my_palette = plt.cm.get_cmap('Set2', len(df_2.index))

    # loop to plot
    for row in range(0, len(df_2.index)):
        make_spider(df = df_2, row=row, title=df_2['player'][row], color=my_palette(row))
    return plt, df_2

    
def predict_playstyle(df_original, model):
    df = df_original.copy()
    try:
        df.drop(columns = 'goals_against_while_last_defender', inplace = True)
    except KeyError:
        pass
    
    to_drop = [
        'shots_against',
        'goals_against',
        'shooting_percentage',
        'bpm',
        'amount_stolen_big',
        'amount_stolen_small',
        'count_collected_big',
        'count_collected_small',
        'count_stolen_small',
        'count_stolen_big',
        'amount_overfill_stolen',
        'time_zero_boost',
        'time_full_boost',
        'time_boost_0_25',
        'time_boost_25_50',
        'time_boost_50_75',
        'time_boost_75_100',
        'avg_speed',
        'total_distance',
        'time_supersonic_speed',
        'time_boost_speed',
        'time_slow_speed',
        'time_ground',
        'time_low_air',
        'time_high_air',
        'time_powerslide',
        'time_defensive_third',
        'time_neutral_third',
        'time_offensive_third',
        'time_defensive_half',
        'time_offensive_half',
        'time_behind_ball',
        'time_infront_ball',
        'time_most_back',
        'time_most_forward',
        'time_closest_to_ball',
        'time_farthest_from_ball',
        'mvp',
        'taken',
        'goals',
        'assists',
        'avg_distance_to_ball_possession',
        'shots',
        'amount_stolen'
    ]
    
    df.drop(columns = to_drop, inplace = True)
    df.drop(columns = ['player_name'], inplace = True)
    for col in df.columns[:-1]:
        df[col] = df[col].astype(float)
        
    # df = poly.transform(ss.transform(df))
    df_preds = pd.DataFrame(model.predict_proba(df), columns = ['MonkeyMoon', 'Oski', 'Vatira'])       
    df_player= model.predict(df)
    return df_preds, df_player

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

    

def predict_rank(df_original, model, dict_ranks):
    df = df_original.copy()
    
    to_drop = [
        'shots_against',
        'goals_against',
        'shooting_percentage',
        'bpm',
        'amount_stolen_big',
        'amount_stolen_small',
        'count_collected_big',
        'count_collected_small',
        'count_stolen_small',
        'count_stolen_big',
        'amount_overfill_stolen',
        'time_zero_boost',
        'time_full_boost',
        'time_boost_0_25',
        'time_boost_25_50',
        'time_boost_50_75',
        'time_boost_75_100',
        'avg_speed',
        'total_distance',
        'time_supersonic_speed',
        'time_boost_speed',
        'time_slow_speed',
        'time_ground',
        'time_low_air',
        'time_high_air',
        'time_powerslide',
        'time_defensive_third',
        'time_neutral_third',
        'time_offensive_third',
        'time_defensive_half',
        'time_offensive_half',
        'time_behind_ball',
        'time_infront_ball',
        'time_most_back',
        'time_most_forward',
        'time_closest_to_ball',
        'time_farthest_from_ball',
        'mvp',
        'taken',
        'goals',
        'assists',
        # 'avg_distance_to_ball_possession',
        'shots',
        # 'amount_stolen',
        'saves',
        'score'
    ]
    
    df.drop(columns = to_drop, inplace = True)
    try:
        df.drop(columns = 'Unnamed: 0', inplace = True)
    except KeyError:
        pass
    
    try:
        df.drop(columns = 'goals_against_while_last_defender', inplace = True)
    except KeyError:
        pass
    
    try:
        df.drop(columns = 'player_name', inplace = True)
    except KeyError:
        pass
    rank_preds = model.predict(df)
    rank_preds_text = [dict_ranks[int(pred)] for pred in rank_preds]
    return rank_preds_text
    