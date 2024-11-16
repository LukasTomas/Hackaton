import os
import time
import pandas as pd
from model import Model

from environment import Environment


class Evaluator:

    DEFAULT_TEST_GAMES = os.path.join('..', 'data', 'games.csv') 
    DEFAULT_TEST_PLAYERS = os.path.join('..', 'data', 'players.csv')

    def __init__(self, games_percent=0.2, test_games=DEFAULT_TEST_GAMES, test_players=DEFAULT_TEST_PLAYERS):
        self.games = pd.read_csv(test_games, index_col=0)  
        self.games["Date"] = pd.to_datetime(self.games["Date"])
        self.games["Open"] = pd.to_datetime(self.games["Open"])

        games_number = int(self.games.shape[0] * games_percent)
        self.games = self.games.iloc[:games_number]

        self.players = pd.read_csv(test_players, index_col=0)                                                 # TODO change .. to .
        self.players["Date"] = pd.to_datetime(self.players["Date"])

    def evaluate(self, nn):
        env = Environment(self.games, self.players, Model(nn), 
                          init_bankroll=1000, min_bet=5, max_bet=100)

        evaluation = env.run()
        print(f'Final bankroll: {env.bankroll:.2f}')
        return env.bankroll


