import torch
import pandas as pd
import numpy as np

class DataPreprocessing:
    def GetTensor(self, team_A_data, team_H_data):
        A_data = self.Processing(team_A_data)
        H_data = self.Processing(team_H_data)

        Tensor = self.Tenzor(A_data, H_data)
        return Tensor                                                               # return torch.Tensor

    def Processing(self, data):
        data = pd.DataFrame(data)
        data = self.Change_Date(data)
        return data

    def Tenzor(self, data_team1, data_team2):
        data_team1 = pd.DataFrame(data_team1)
        data_team2 = pd.DataFrame(data_team2)
        data = pd.concat([data_team1, data_team2], ignore_index=True)
        data = data.to_numpy().flatten().astype(np.float32)
        return torch.tensor(data)

    def Change_Date(self, data_to_transform):                                       # Changing Date to int (in the process year is lost)
        data_to_transform["Day"] = pd.NA
        data_to_transform["Month"] = pd.NA
        for index ,row in data_to_transform.iterrows():
            Date = row["Date"]
            Digits = ((str)(Date)).split("-")
            LastDigits = Digits[2].split(" ")
            data_to_transform.loc[index, ["Day"]] = int(LastDigits[0])
            data_to_transform.loc[index, ["Month"]] = int(Digits[1])
        data_to_transform.drop(columns=["Date"], inplace=True)
        return data_to_transform