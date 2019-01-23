import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inputFile = "D:/Research/Result/checkin/topic_doc_dis_1540973167751.csv"
data = pd.read_csv(inputFile, sep=' ', header=None)
# print(data)
# print(data.values[0])

places_users_file = "D:/Research/Dataset/checkin/user_checkin_above_10x10x5_us_train - Copy.txt"
places_users = []
with open(places_users_file, "r") as inFile:
    for line in inFile:
        places_users.append(list(map(int, line.rstrip().split(" "))))
# print(places_users)

places_users_test_file = "D:/Research/Dataset/checkin/user_checkin_above_10x10x5_us_test - Copy.txt"
places_users_test = []
with open(places_users_test_file, "r") as inFile:
    for line in inFile:
        places_users_test.append(list(map(int, line.rstrip().split(" "))))
# print(places_users)

diss = []
# i = 0
dis_sum = 0.0
threshold = 0.1
under_threshold_count = 0
index = 6
places_similar_user = []
i = 0
match_count = 0
for user in data.values:
    distance = np.linalg.norm(user-data.values[index])
    dis_sum += distance
    diss.append(distance)
    if distance < threshold:
        if i != index:
            under_threshold_count += 1
            places_similar_user.extend(places_users[i])
        # if distance == 0:
        #     match_count += 1
    i += 1

print(len(set(places_similar_user)))
print(len(set(places_users_test[index])))
print(len(set(places_similar_user) & set(places_users_test[index])))
# print(match_count)

print(dis_sum / (len(data.values)-1))
# print(under_threshold_count)
# print(len(diss))
# print(len(diss[1:]))

# n, bins, patches = plt.hist(diss, 50, density=False, facecolor='g', alpha=0.75)
# plt.grid(True)
# plt.show()
