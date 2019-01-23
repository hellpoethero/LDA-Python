import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

inputFile = "D:\Research\Dataset\checkin/Gowalla_totalCheckins_chekin10.txt"
data = pd.read_csv(inputFile, sep='\t', header=None)
print(len(data))

print(data)
x1 = data.groupby([0])[1]
x1_list = x1.apply(list)
# print(x1_list)
# print(len(x1_list[0]))

checkin_count = []
unique_checkin_count = []
train_unique_place_stats = []
b = []
avg_ratios = []
for i in range(0, 200):
    avg_ratios.append([])
    b.append(i+1)
users = []
for user in x1_list:
    # users.append(len(user))
    if len(user) >= 10:
        # users.append(len(set(user)))
        test = user[:round(len(user) * 0.2)]
        train = user[round(len(user) * 0.2):]
        test_set = set(test)
        train_set = set(train)
        intersection = test_set & train_set
        checkin_count.append(len(train))
        unique_checkin_count.append(len(train_set))
        # print(len(train_set))
        train_unique_place_stats.append(len(train_set))
        if len(train_set) < 200:
            avg_ratios[len(train_set)].append(float(len(intersection)) / len(test_set))
        # if len(train_set) < 4:
        #     users.append(float(len(intersection)) / len(test_set))
    #     print(str(len(test_set)) + " " + str(len(train_set)) + " " + str(len(intersection)))

a = []
for avg_ratio in avg_ratios:
    a.append(np.average(avg_ratio))

# the histogram of the data
# n, bins, patches = plt.hist(train_unique_place_stats, 500, density=False, facecolor='g', alpha=0.75)
plt.grid(True)
# plt.show()

# plt.plot(np.array(checkin_count), np.array(unique_checkin_count), 'ro')
plt.plot(np.array(b), np.array(a), 'ro')
plt.show()
