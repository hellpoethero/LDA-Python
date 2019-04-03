import pandas as pd
import random

# inputFile = "D:\Research\Dataset\Gowalla_totalCheckins.txt"
inputFile = "D:/Research/Dataset/checkin/Gowalla_totalCheckins_chekin10.txt"
data = pd.read_csv(inputFile, sep='\t', header=None)

us_ca = pd.read_csv("D:/Research/Dataset/checkin/sanfrancisco.txt", sep=',', header=0)
us_ca_places = us_ca['id'].values.tolist()
# print(us_ca_places)
# print(data)

a = data[data[1].isin(us_ca_places)]
# a = data
print(a)
print(len(a))
print(a[1].value_counts())

x1 = a.groupby(0)[1]
x1_list = x1.apply(list)
# print(x1_list)

train_ratio = 0.8
places = []
places_train = []
places_test = []
places_validation = []
train_count = 0
test_count = 0
validation_count = 0
train_set = set()
test_set = set()
validation_set = set()
for place in x1_list:
    if len(place) > 5 and len(set(place)) >= 2:
        places_test.append("a b "+str(place[-1]))
        test_count += 1
        place = place[:-1]
        random.shuffle(place)
        train = []
        validation = []
        for p in place:
            if random.uniform(0.01, 1.0) <= train_ratio \
                    and len(train) < len(place) * 0.8:
                train.append(p)
                train_count += 1
                train_set.add(p)
            else:
                validation.append(p)
                validation_count += 1
                validation_set.add(p)
        # place_str = " ".join(map(str, place))
        places_train.append("a b "+" ".join(map(str, train)))
        places_validation.append("a b "+" ".join(map(str, validation)))

print(len(places_train))
print(len(places_validation))
print(train_count)
print(validation_count)
print(len(train_set))
print(len(validation_set))
print(len(train_set & validation_set))

with open("D:/Research/Dataset/checkin/sanfrancisco/user_checkin_5x2_test.txt", "w") as outFile:
    outFile.write("\n".join(places_test))

with open("D:/Research/Dataset/checkin/sanfrancisco/user_checkin_5x2_train.txt", "w") as outFile:
    outFile.write("\n".join(places_train))

with open("D:/Research/Dataset/checkin/sanfrancisco/user_checkin_5x2_validation.txt", "w") as outFile:
    outFile.write("\n".join(places_validation))

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
