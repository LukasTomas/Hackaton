import numpy as np
import pandas as pd
from Database import Database
from Strategies.trading_strategy_kelly_exp_val import strat_kelly_exp_value as Strat
import DataProcessing
import torch
import nn
from hyperparams import INPUT_MATCH_COUNT
import random

class Model:
    def __init__(self, neuralNetwork=None):
        self.database = Database()
        # self.database.Inicialization()
        self.data_processing = DataProcessing.DataPreprocessing()

        if neuralNetwork is None:
            self.model = nn.LinearNN()
            self.model.load_state_dict(torch.load("best_nn.pth"))
        else:
            self.model = neuralNetwork

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        self.database.UpdateGames(inc[0])
        opps = self.Gen_our_probability(opps)
        opps = self.Modify_opps_for_strategie(opps)
        bets = Strat(summary, opps)
        bets = self.Modify_bets(bets)
        # bets = pd.DataFrame(columns=["BetH", "BetA"])
        return bets
    
    def Gen_our_probability(self, opps):
        opps["ProH"] = pd.NA
        for index, row in opps.iterrows():
            HID = row["HID"]
            AID = row["AID"]
            Team_H_data = self.database.Return_team_data(HID)
            Team_A_data = self.database.Return_team_data(AID)
            # Team_H_data = self.database.Return_team_data(HID)
            # Team_A_data = self.database.Return_team_data(AID)

            if ((len(Team_A_data) == INPUT_MATCH_COUNT) and (
                    len(Team_H_data) == INPUT_MATCH_COUNT)):  # TODO Not sure with the shape[index]
                # tensor = self.data_processing.GetTensor(Team_A_data, Team_H_data)
                int_list = []
                for team in [Team_A_data, Team_H_data]:
                    for series in team:
                        int_list.extend(series.values)
                tensor = torch.tensor(int_list).float()
                our_prob_H = self.model(tensor).item()
                # our_prob_H = random.uniform(0, 1)
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
        bets.drop(columns=["Date", "OddsH", "OddsA", "BetH", "BetA", "ProH"], inplace=True)
        bets.rename(columns = {"NewBetH" : "BetH", "NewBetA" : "BetA"}, inplace=True)
        return bets