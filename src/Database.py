from hyperparams import GAMES_FILE, PLAYERS_FILE, STACK_SIZE, INPUT_MATCH_COUNT
from collections import deque
import pandas as pd
import time
class Database:
    def __init__(self):
        self.Games = []
        self.Teams_dictionary = {}
        self.Incrementor = 0                                            # Makes sure that each game have its own ID
    def Inicialization(self):                                           # Inicialization on GAMES_FILE
        filecontent = pd.read_csv(GAMES_FILE)
        filecontent.drop(columns=["Unnamed: 0"], inplace=True)
        for index, row in filecontent.iterrows():
            self.Games.append(row)

            HID = row["HID"]
            self.Dictionary_Update(HID)
            AID = row["AID"]
            self.Dictionary_Update(AID)

            self.Team_Update(HID)
            self.Team_Update(AID)
            self.Incrementor += 1

    def UpdateGames(self, data):
        content = pd.DataFrame(data)                                                   # TODO
        for index, row in content.iterrows():
            self.Games.append(row)

            HID = row["HID"]
            self.Dictionary_Update(HID)
            AID = row["AID"]
            self.Dictionary_Update(AID)

            self.Team_Update(HID)
            self.Team_Update(AID)

            self.Incrementor += 1

    def Dictionary_Update(self, Team_ID):                               # Checks if the team exists in dic
        if Team_ID not in self.Teams_dictionary:
            self.Teams_dictionary[Team_ID] = Team(Team_ID)

    def Team_Update(self, Team_ID):
        self.Teams_dictionary[Team_ID].Add_match(self.Incrementor)

    def Return_team_data(self, Team_ID):
        self.Dictionary_Update(Team_ID)
        Games_Ids = self.Teams_dictionary[Team_ID].Return_last_x_matches_id(INPUT_MATCH_COUNT)
        Games_Ids.reverse()                                             # Makes sure you go from newest to oldest games
        data_frames = [self.Games[GameId] for GameId in Games_Ids]

        if data_frames:
            data_frame = pd.concat(data_frames, ignore_index=True)
        else:
            data_frame = pd.DataFrame()
        return data_frame

class Team:
    def __init__(self, Id):
        self.Id = Id
        self.Games_history = deque()

    def Add_match(self, game):
        self.Games_history.append(game)
        if (self.History_size() > STACK_SIZE):
            self.Remove_match()

    def Remove_match(self):
        self.Games_history.popleft()

    def History_size(self):
        return len(self.Games_history)

    def Return_last_x_matches_id(self, x):
        return list(self.Games_history)[-x:]