Project 2 readme

python3.6.2
packages used - pandas, re, os, numpy, matplotlib, torch, gym

The LunarLander.py contains hyperparamater ranges that can be tuned to train agents.
Running LunarLander generates DQNAgents (contained in DQNAgent.py) and saves the scores of the training results in the data folder.
It saves the models in the models folder if they are able to achieve a score of 200.
Grapher.py can be used to produce plots that read the score data from data folder.
TestModel.py can be used to evaluate the performance of saved models and generate score data.
