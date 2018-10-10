import pandas as pd
import random

# inputFile = "D:\Research\Dataset\Gowalla_totalCheckins.txt"
inputFile = "D:/Research/Project/LDA/data/checkin/Gowalla_totalCheckins_chekin10.txt"
data = pd.read_csv(inputFile, sep='\t', header=None)

print(len(data))

x1 = data.groupby(0)[1]
x1_list = x1.apply(list)
# print(x1_list)

train_ratio = 0.8
places = []
places_train = []
places_test = []
train_count = 0
test_count = 0
train_set = set()
test_set = set()
for place in x1_list:
    if len(place) >= 10 and len(set(place)) >= 5:
        random.shuffle(place)
        train = []
        test = []
        for p in place:
            if random.uniform(0.01, 1.0) <= train_ratio \
                    and len(train) < len(place) * 0.8:
                train.append(p)
                train_count += 1
                train_set.add(p)
            else:
                test.append(p)
                test_count += 1
                test_set.add(p)
        # place_str = " ".join(map(str, place))
        places_train.append(" ".join(map(str, train)))
        places_test.append(" ".join(map(str, test)))

print(len(places_train))
print(len(places_test))
print(train_count)
print(test_count)
print(len(train_set))
print(len(test_set))
print(len(train_set & test_set))

# with open("D:/Research/Project/LDA/data/checkin/user_checkin_above_10x10x5_train.txt", "w") as outFile:
#     outFile.write("\n".join(places_train))
#
# with open("D:/Research/Project/LDA/data/checkin/user_checkin_above_10x10x5_test.txt", "w") as outFile:
#     outFile.write("\n".join(places_test))

# x2 = data.groupby(1)[0]
# print(len(x2))
# print(x2.apply(list))
# x2_count = x2.agg('count')
# result = []
# for index, row in data.iterrows():
#     if x2_count.loc[row[4]] >= 10:
#         result.append([row[0], row[4]])
#         print(len(result))
#     # if index > 10000:
#     #     break
#
# print("----------")
# print(len(result))
#
# with open("D:/Research/Project/LDA/data/checkin/Gowalla_totalCheckins_chekin10.txt", "w") as outFile:
#     for line in result:
#         outFile.write(str(line[0]) + "\t" + str(line[1]) + "\n")
