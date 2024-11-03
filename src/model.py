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
        self.database.Inicialization()
        self.data_processing = DataProcessing.DataPreprocessing()
        self.model = nn.LinearNN()
        self.model.load_state_dict(torch.load("best_nn.pth"))
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        self.database.UpdateGames(inc[0])
        opps = self.Gen_our_probability(opps)
        opps = self.Modify_opps_for_strategie(opps)
        bets = Strat(summary, opps)
        bets = self.Modify_bets(bets)
        return bets
    def Gen_our_probability(self, opps):
        opps["ProH"] = pd.NA
        for index, row in opps.iterrows():
            HID = row["HID"]
            AID = row["AID"]
            Team_H_data = pd.DataFrame(self.database.Return_team_data(HID))
            Team_A_data = pd.DataFrame(self.database.Return_team_data(AID))
            if ((Team_A_data.shape[0] == INPUT_MATCH_COUNT) and (
                    Team_H_data.shape[0] == INPUT_MATCH_COUNT)):  # TODO Not sure with the shape[index]
                tensor = self.data_processing.GetTensor(Team_A_data, Team_H_data)
                our_prob_H = self.model(tensor).item()
                opps.loc[index, ["ProH"]] = our_prob_H
        return opps

    def Modify_opps_for_strategie(self, opps):
        opps.drop(columns=["Season"], inplace=True)
        opps.drop(columns=["HID"], inplace=True)
        opps.drop(columns=["AID"], inplace=True)
        opps.drop(columns=["N"], inplace=True)
        opps.drop(columns=["POFF"], inplace=True)
        return opps

    def Modify_bets(self, bets):
        # TODO
        bets.drop(columns=["Date", "OddH", "OddsA", "BetH", "BetA", "ProH"], inplace=True)
        bets.rename(columns = {"NewBetH" : "BetH", "NewBetA" : "BetA"}, inplace=True)
        return bets