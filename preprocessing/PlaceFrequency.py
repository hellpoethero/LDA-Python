import ReadData
import SortedWord
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# inputFile = "D:/Research/Dataset/checkin/Gowalla_totalCheckins_chekin10.txt"
# data = pd.read_csv(inputFile, sep='\t', header=None)

data = ReadData.ReadData()
trainPath = "D:\Research\Dataset\checkin/user_checkin_above_10x10x5_sf_train - Copy.txt"
train = data.read_train(trainPath)

unique_user_place_counts = data.get_users_places().groupby(['user', 'place']).size().groupby(['place']).size()
place_list = unique_user_place_counts.index.tolist()
place_count_list = unique_user_place_counts.values.tolist()
unique_user_place_counts.to_csv('D:/Research/Result/checkin/place_unique_user_sf.csv', index=True)
