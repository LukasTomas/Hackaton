import os
import pandas as pd

DATA_DIR = os.path.join('..', 'data')
PLAYERS_FILE = os.path.join(DATA_DIR, 'players.csv')
GAMES_FILE = os.path.join(DATA_DIR, 'games.csv')
for f in [PLAYERS_FILE, GAMES_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f)

games = pd.read_csv(GAMES_FILE)

INPUT_MATCH_COUNT = 5                       # Number of input matches for neural network
MATCH_DATAPOINT_COUNT = games.shape[1] - 1  # Number of datapoints per match; -1 to exclude 'Unnamed: 0' column

INPUT_TEAM_COUNT = INPUT_MATCH_COUNT * MATCH_DATAPOINT_COUNT
NN_INPUT_SIZE = INPUT_TEAM_COUNT * 2

STACK_SIZE = 50

# hyperparameters for evaluation
GENERATIONS = 50
POPULATION_SIZE = 20
MUTATION_PROB = 0.1