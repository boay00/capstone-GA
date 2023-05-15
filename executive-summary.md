# Executive Summary 
---

Rocket League is a highly mechinical player vs player video game, requiring high muscle memory, concentration and dedication in order to achieve mastery of the principles and underlying mechanics. To a lay viewer, high level gameplay looks absurdly complex, and even more so to a begginner of the game. Despite this, the initial skill curve is relatively shallow in gradient, and players can often climb through low ranks in a short period of time. The plateau however arrives at mid to higher ranks. This is the point in which core principles have been learnt, however the route to improvement becomes unclear.

A mixture of advice is available in order to improve, however one piece of advice cannot fit the population of players. Understanding the flaws in a players own gameplay is the most efficient way to target areas to improve. For that reason, we have designed an application, containing machine learning models and mathmatical algorithms, capable of describing gameplay stats in a more digestible way. Given replay data from the website [ballchasing.com], a player can access the raw stats from a game. Furthermore, a playstyle plot can be observed to determine weak areas of gameplay. A  model consisting of an Extra Trees Classification model algorithm was used to assign this playstyle to one of three professional Rocket League Players:
- M0nkey M00n
- Oski
- Vatira
Upon comparing their playstyle with these pro players, the user can then decide which attributes to work on, using the professional players as foundations for improvement.

Furthermore, a regression model can predict the current level of the users gameplay - indicating whether they are performing better or worse than expected, given the available stats. With this information, any mid-high ranked rocket league player will benefit from these key insights into their gameplay.

Whilst this application is a great tool and offers useful information, there are several ways in which it can be improved.
1. Currently, taking one replay at a time is lighter on the workload of the host website, however does not always represent a players playstyle accurately - given the short timescale of a rocket league game.
2. Given greater memory and available resources, stronger models could be created, and further features could be deployed on the application
3. Whilst this application can predict 3v3 gameplay effectively, it is not trained on 2v2 or 1v1 data, which are both popular game modes. Training new models on these games would add to the potential of this application.