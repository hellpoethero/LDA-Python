import ReadData
import SortedWord
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# inputFile = "D:/Research/Dataset/checkin/Gowalla_totalCheckins_chekin10.txt"
# data = pd.read_csv(inputFile, sep='\t', header=None)

data = ReadData.ReadData()
trainPath = "D:\Research\Dataset\checkin/sanfrancisco/user_checkin_5x2_train - Copy.txt"
train = data.read_train(trainPath)

unique_user_place_counts = data.get_users_places().groupby(['user', 'place']).size().groupby(['place']).size()
place_list = unique_user_place_counts.index.tolist()
place_count_list = unique_user_place_counts.values.tolist()
unique_user_place_counts.to_csv('D:/Research/Dataset/checkin/sanfrancisco/place_unique_user_5x2.csv', index=True)
