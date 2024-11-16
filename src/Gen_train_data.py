import pandas as pd
import csv
import os
from hyperparams import GAMES_FILE, PLAYERS_FILE

train_data_size = 0.8

folder = "Test_data"
train_file = os.path.join(folder, "train_file.csv")
test_file = os.path.join(folder, "test_file.csv")

def Creating_datasets():
    filecontent = pd.read_csv(GAMES_FILE)
    train_data, test_data = Data_seperation(filecontent)
    output = filecontent.head(0).columns

    Delete_files_if_exists()
    WriteOutPut(train_file, train_data, output)
    WriteOutPut(test_file, test_data, output)

def WriteOutPut(OutputFile, data, output):
    with open(OutputFile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(output)
        writer.writerows(data)

def Delete_files_if_exists():
    if os.path.exists(train_file):
        os.remove(train_file)  # Delete the file
    if os.path.exists(test_file):
        os.remove(test_file)  # Delete the file

def Data_seperation(filecontent):
    train_data = []
    test_data = []
    size = filecontent.shape[0]
    seperation_value = size * train_data_size
    for index, row in filecontent.iterrows():
        if (seperation_value > index):
            train_data.append(row)
        else:
            test_data.append(row)

    return train_data, test_data

Creating_datasets()