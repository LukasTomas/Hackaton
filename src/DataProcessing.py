import torch
import pandas as pd
import numpy as np

class DataPreprocessing:
    def GetTenzor(self, team_A_data, team_H_data):
        A_data = self.Processing(team_A_data)
        H_data = self.Processing(team_H_data)

        Tenzor = self.Tenzor(A_data, H_data)
        return Tenzor                                                               # return torch.Tensor

    def Processing(self, data):
        data = pd.DataFrame(data)
        data = self.Change_Date(data)
        data = self.Destroy_Open(data)
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
            Digits = Date.split("-")
            data_to_transform.loc[index, ["Day"]] = int(Digits[2])
            data_to_transform.loc[index, ["Month"]] = int(Digits[1])
        data_to_transform.drop(columns=["Date"], inplace=True)
        return data_to_transform

    def Destroy_Open(self, data_to_transform):
        data_to_transform.drop(columns=["Open"], inplace=True)                      # Other date which tells when the betting has been opened (i guess?)
        return data_to_transform