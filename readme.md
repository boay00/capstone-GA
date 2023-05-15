# README.md

---

## Bede Young - GA Capstone Project
---
## Rocket League - Playstyle / Rank Analysis and Prediction

---

![Pro Players Playstyles]('2-cleaning-EDA/code/pros-playstyle.png)

---

### Problem Statement


Rocket league is regarded as a video game with very slow improvement rates. Whilst easy to pick up and enjoy, the highly mechanical and precise nature of the game leads to very slow progression at high ranks. Professional e-sports players are constantly pushing the skill ceiling in terms of mechanical talent and tactical mastery, and can be used as examples for improvement and furthermore as play-style case studies for developing players, particularly at mid to high ranks.
This project uses 3 pro players, selected as top-level examples of 3 distinct play-styles:
- MonkeyMoon: 
    - Highly efficient with boost, never out of the game because of this
    - Very strong decision making - only uses the required amount of resources to make a favourable play for his team
- Vatira: 
    - ‘3rd man’ position, largely playing with a high amount of boost, able to use his ability to capitalise on space created by teammates
    - Plays close to teammates and pressure the opposition collectively
- Oski: 
    - ‘Ballchaser’, uses a large amount of boost to play as fast as possible, aiming to beat opposition to the ball or take out multiple players in one play
    - Often leaves teammates exposed, however looks to create more opportunities than conceded over a game
---
An ExtraTrees classification model has been trained on replay data from professional tournaments for each of these players, and can be used to predict which play-style a player most closely fits from a submitted 3v3 replay ID and player ID.
- Visuals highlighting key differences to each play-style can also be viewed, as well as the probabilities of each play-style being selected.
---
Furthermore, a neural network regression model has been produced to estimate the rank of a player given the stats from a 3v3 replay ID and player ID

---
## Data Dictionary

|Feature|Type|Description|
|-|-|-|
|shots|int|the total number of shots for **player_name** in a match|
|shots_against|int|the total number of shots against the **player_name** team in a match|
|goals|int|the total number of goals for **player_name** in a match|
|goals_against|int|the total number of goals against the **player_name** team in a match|
|saves|int|the total number of saves for **player_name** in a match|
|assists|int|the total number of assists for **player_name** in a match|
|score|int|the total score for **player_name** in a match|
|mvp|boolean|True if **player_name** won the match **and** received the most score on the winning team|
|shooting_percentage|float|the percentage of shots that lead to a goal for **player_name** in a match|
|bcpm|float|the amount of boost consumed per minute for **player_name** in a match|
|avg_amount|float|the average boost amount for **player_name** throughout the match|
|amount_collected|float|the total amount of boost collected by **player_name** in a match|
|amount_stolen|float|the total amount of boost stolen by **player_name** in a match|
|amount_collected_big|float|the total amount of large boost pads collected by **player_name** in a match|
|amount_stolen_big|float|the total amount of large boost pads stolen by **player_name** in a match|
|amount_collected_small|float|the total amount of small boost pads collected by **player_name** in a match|
|amount_stolen_small|float|the total amount of small boost pads stolen by **player_name** in a match|
|count_collected_big|int|the total count of big boost pads collected by **player_name** in a match|
|count_stolen_big|int|the total count of big boost pads stolen by **player_name** in a match| 
|count_collected_small|int|the total count of small boost pads collected by **player_name** in a match|
|count_stolen_small|int|the total count of small boost pads stolen by **player_name** in a match|
|amount_overfill|float|the total amount of boost collected which was lost due to the cap at 100 boost|
|amount_overfill_stolen|float|the total amount of boost stolen which was lost due to the cap at 100 boost|
|amount_used_while_supersonic|float|the total amount of boost used while **player_name** was already travelling at maximum speed|
|time_zero_boost|float|the total time in seconds **player_name** spent at 0 boost in a match|
|percent_zero_boost|float|the percentage of time **player_name** spent at 0 boost in a match|
|time_full_boost|float|the total time in second **player_name** spent at 100 boost in a match|
|percent_full_boost|float|the percentage of time **player_name** spent at 100 boost in a match|
|time_boost_0_25|float|the total time in seconds **player_name** spent at between 0 and 25 boost|
|time_boost_25_50|float|the total time in seconds **player_name** spent at between 25 and 50 boost|
time_boost_50_75|float|the total time in seconds **player_name** spent at between 50 and 75 boost|
time_boost_75_100|float|the total time in seconds **player_name** spent at between 75 and 100 boost|
|percent_boost_0_25|float|the percentage of time **player_name** spent at between 0 and 25 boost|
|percent_boost_25_50|float|the percentage of time **player_name** spent at between 25 and 50 boost|
|percent_boost_50_75|float|the percentage of time **player_name** spent at between 50 and 75 boost|
|percent_boost_75_100|float|the percentage of time **player_name** spent at between 75 and 100 boost|
|avg_speed|float|average speed of **player_name** in au|
|total_distance|float|total distance covered by **player_name** in a match|
|time_supersonic_speed|float|the total time in seconds **player_name** spent at supersonic speed|
|time_boost_speed|float|the total time in seconds **player_name** spent boosting|
|time_slow_speed|float|the total time in seconds **player_name** spent at slow speed|
|time_ground|float|the total time in seconds **player_name** spent on the ground|
|time_low_air|float|the total time in seconds **player_name** spent  low in the air|
|time_high_air|float|the total time in seconds **player_name** spent high in the air|
|time_powerslide|float|the total time in seconds **player_name** spent powersliding|
|count_powerslide|int|the total number of times **player_name** pressed the powerslide button in a match|
|avg_powerslide_duration|float|the average time in seconds the player held down the powerslide button in a match|
|avg_speed_percentage|float|the average speed of **player_name** as a percentage of full speed|
|percent_slow_speed|float|the percentage of time **player_name** spent at slow speed|
|percent_boost_speed|float|the percentage of time **player_name** spent boosting|
|percent_supersonic_speed|float|the percentage of time **player_name** spent at supersonic (maximum) speed|
|percent_ground|float|the percentage of time **player_name** spent on the ground|
|percent_low_air|float|the percentage of time **player_name** spent low in the air|
|percent_low_air|float|the percentage of time **player_name** spent high in the air|
|avg_distance_to_ball|float|the average_distance of **player_name** to the ball in AU in a match|
|avg_distance_to_ball_possession|float|the average_distance of **player_name** to the ball in AU in a match while **player_name** team has possession|
|avg_distance_to_ball_no_possession|float|the average_distance of **player_name** to the ball in AU in a match while **player_name** team does not have possession|
|avg_distance_to_mates|float|the average distance of **player_name** to teammates in a match|
|time_defensive_third|float|the total time in seconds **player_name** spent in the defensive third of the pitch|
|time_neutral_third|float|the total time in seconds **player_name** spent in the neutral third of the pitch|
|time_offensive_third|float|the total time in seconds **player_name** spent in the offensive third of the pitch|
|time_defensive_half|float|the total time in seconds **player_name** spent in the defensive half of the pitch|
|time_offensive_half|float|the total time in seconds **player_name** spent in the offensive half of the pitch|
|time_behind_ball|float|the total time in seconds **player_name** spent behind the ball|
|time_infront_ball|float|the total time in seconds **player_name** spent in front of the ball|
|time_most_back|float|the total time in seconds **player_name** spent as the furthest player back on the team|
|time_most_forward|float|the total time in seconds **player_name** spent as the most forward player on the team|
|goals_against_while_last_defender|int|the total number of goals conceeded by **player_name** while the furthest player back|
|time_closest_ball|float|the total time in seconds **player_name** spent as the player closest to the ball|
|time_farthest_ball|float|the total time in seconds **player_name** spent as the player farthest from the ball|
|percent_defensive_third|float|the percent of time **player_name** spent in the defensive third|
|percent_offensive_third|float|the percent of time **player_name** spent in the offensive third|
|percent_neutral_third|float|the percent of time **player_name** spent in the neutral third|
|percent_defensive_half|float|the percent of time **player_name** spent in the defensive half|
|percent_offensive_third|float|the percent of time **player_name** spent in the offensive half|
|percent_behind_ball|float|the percent of time **player_name** spent behind the ball in a match|
|percent_infront_ball|float|the percent of time **player_name** spent infront of the ball in a match|
|percent_most_back|float|the percent of time **player_name** spent as the player furthest back in a match|
|percent_most_forward|float|the percent of time **player_name** spent as the player most forward in a match|
|percent_closest_to_ball|float|the percent of time **player_name** spent as the player closest to the ball in a match|
|percent_farthest_to_ball|float|the percent of time **player_name** spent as the player farthest from the ball in a match|
|inflicted|int|the total number of demolitions **player_name** inflicted on the opposition in a match|
|taken|int|the total number of demolitions **player_name** suffered from the opposition in a match|
|player_name|string|the gamertag of the player in the match|

---

### Conclusions and Recommendations

The playstyle algorithm was successful in identifying key attributes of the three pro players' playstyles. Monkey Moon was shown to be a highly efficient and strong team player, despite slightly slower speed and aggression stats. Oski was calculated to have a fast paced, aggressive play style, whilst lacking strong boost efficiency in comparison to the other players. Vatira was found to have a well rounded playstyle, scoring highly in all attributes.

This algorithm was somewhat successful in predicting other players' playstyle. Given a single replay it is unable to provide a reliable estimate of a playstyle. In future, building the app to allow multiple replays to be analysed would be a good way to improve this feature. This would be slightly complex, and may discourage casual users from testing the app, since the user would most likely need to generate a personal authorisation api token to perform this. Furthermore, this feature would cause strain on the website being scraped, so perhaps this is something to keep in mind for the future.

Classifying a user to one on the three professional players' playstyles was also somewhat successful. Expected predictions were observed in many cases, however - occasional exceptions were seen, in which the predicted playstyle did not match the shape of the pro players playstyle at all. Furthermore, Oski's playstyle was rarely selected as the best match, particuarly in low ranked players. This is likely due to Oski having a very hard to replicate playstyle for low ranked players.

Rank predictions were successful in estimating the approximate rank. The aim was not to produce a model which predicted the correct rank 100% of the time, but to highlight the level of the performance in relation to the rank the replay was from. For example, a player in Champion could use the predictor to assess whether their stats allign with the prediction of Champion rank players.

### Other Recommendations

- Using other pro players as playstyle examples would potentially incorporate more playstyles and make the final application even more useful as a reference for improvement

- training a model on only high ranked players (diamond and higher), would more accurately place players in their ranks.
    - Because of the barrier to uploading replays to ballchasing.com, only players already experienced are likley to upload replays - therefore the majority of low ranked replays are likely strong players using low ranked accounts to play with friends. This may have negatively impacted the models performance
    
---

### File Structure

├── 1-data-collection
│   ├── code
│   │    ├── capstone-data-collection.ipynb
│   │    ├── data-collection-friends.ipynb
│   │    ├── random_replays.ipynb
│   ├── data
│   │    ├── all_replays.csv
│   │    ├── boay00.csv
│   │    ├── cho_shmo.csv
│   │    ├── eden.csv
│   │    ├── monkeymoon.csv
│   │    ├── oski.csv
│   │    ├── vati.csv
│   │    ├── vinu.csv
├── 2-cleaning-EDA
│   ├── code
│   │    ├── all_players_cleaning.ipynb
│   │    ├── all-ranks-EDA.ipynb
│   │    ├── Application-function-test.ipynb
│   │    ├── max_metrics.pickle
│   │    ├── pro-EDA.ipynb
│   │    ├── pros-cleaning.ipynb
│   │    ├── pros-playstyle.png
│   ├── data
│   │    ├── boay_cleaned.csv
│   │    ├── cho_cleaned.csv
│   │    ├── cleaned_random_replays.csv
│   │    ├── eden_cleaned.csv
│   │    ├── pros.csv
│   │    ├── vinu_cleaned.csv
├── 3-modeling-playstyle
│   ├── code
│   │    ├── playstyle-classification-model.ipynb
│   ├── data
│   │    ├── boay_.csv
├── 3b-playstyle analysis
│   ├── code
│   │    ├── playstyle-analysis-casuals.ipynb
│   ├── data
├── 4-rank-predictor
│   ├── code
│   │    ├── rank-predictions.ipynb
│   ├── data
│   │    ├── boay_for_testing.csv
├── app_files
│   ├── ranks
│   ├── ranks_2
│   ├── dict_ranks.pkl
│   ├── et_model_5.pkl
│   ├── functions_local.py
│   ├── functions.py
│   ├── max_metrics.pickle
│   ├── nn_model_tf_save.h5
│   ├── poly_ps.pkl
│   ├── requirements-Copy1.txt
│   ├── rl_app_copy.py
│   ├── ss_ps.pkl
│   ├── ss.pkl
│   ├── tf_save_model_3.h5
├── executive-summary.md
├── readme.md

