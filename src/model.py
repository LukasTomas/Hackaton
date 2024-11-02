import numpy as np
import pandas as pd
from Database import Database
from Strategies.trading_strategy_kelly_exp_val import strat_kelly_exp_value as Strat
import DataProcessing
import torch
import nn
from hyperparams import INPUT_MATCH_COUNT

class Model:
    def __init__(self):
        self.database = Database()
        # self.database.Inicialization() #Destroy Later
        self.data_processing = DataProcessing.DataPreprocessing()
        self.model = nn.LinearNN()
        self.model.load_state_dict(torch.load("best_nn.pth"))
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        self.database.UpdateGames(inc[0])

        bets = []                                                                                                       # TODO Summary
        for index, row in opps.iterrows():
            HID = row["HID"]
            AID = row["AID"]
            OddsH = row["OddsH"]
            OddsA = row["OddsA"]
            Team_H_data = pd.DataFrame(self.database.Return_team_data(HID))
            Team_A_data = pd.DataFrame(self.database.Return_team_data(AID))
            if ((Team_A_data.shape[0] == INPUT_MATCH_COUNT) and (Team_H_data.shape[0] == INPUT_MATCH_COUNT)):           # TODO Not sure with the shape[index]
                tensor = self.data_processing.GetTensor(Team_A_data, Team_H_data)
                our_prob_H = self.model(tensor).item()
                H_bets, A_bets = Strat(our_prob_H, OddsH, OddsA, 1000)                                          # TODO
                bets.append((H_bets, A_bets))

        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index[:len(bets)])
        return bets
    