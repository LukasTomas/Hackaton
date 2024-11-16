import torch
import pandas as pd
import numpy as np
import time
class DataPreprocessing:
    def GetTensor(self, team_A_data, team_H_data):
        A_data = self.Processing(team_A_data)
        H_data = self.Processing(team_H_data)
        Tensor = self.Tenzor(A_data, H_data)
        return Tensor                                                               # return torch.Tensor

    def Processing(self, data):
        data = pd.DataFrame(data)
        data = self.Destroy_Date(data)
        data = self.Destroy_Open(data)
        return data

    def Tenzor(self, data_team1, data_team2):
        data_team1 = pd.DataFrame(data_team1)
        data_team2 = pd.DataFrame(data_team2)
        data = pd.concat([data_team1, data_team2], ignore_index=True)
        data = data.to_numpy().flatten().astype(np.float32)
        return torch.tensor(data)

    def Destroy_Date(self, data_to_transform):
        if "Date" in data_to_transform.columns:
            data_to_transform.drop(columns=["Date"], inplace=True)
        return data_to_transform

    def Destroy_Open(self, data_to_transform):
        if "Open" in data_to_transform:
            data_to_transform.drop(columns=["Open"], inplace=True)
        return data_to_transform