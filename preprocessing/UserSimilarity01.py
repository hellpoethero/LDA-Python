import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inputFile = "D:/Research/Result/checkin/topic_doc_dis_1541487190337.csv"
data = pd.read_csv(inputFile, sep=' ', header=None)

words = []
places_users_file = "D:/Research/Dataset/checkin/user_checkin_above_10x10x5_us_train - Copy.txt"
places_users = []
with open(places_users_file, "r") as inFile:
    for line in inFile:
        places_users.append(list(map(int, line.rstrip().split(" "))))
        words.extend(list(map(int, line.rstrip().split(" "))))

word_set = list(set(words))
numWord = len(word_set)

places_users_test_file = "D:/Research/Dataset/checkin/user_checkin_above_10x10x5_us_test - Copy.txt"
places_users_test = []
with open(places_users_test_file, "r") as inFile:
    for line in inFile:
        places_users_test.append(list(map(int, line.rstrip().split(" "))))

topic_seqs = []
with open("D:/Research/Result/checkin/topic_seq_1541487190337.csv", "r") as topic_seq_file:
    for line in topic_seq_file:
        topic_seqs.append(list(map(int, line.rstrip().split(" "))))

word_topic_count = [[0] * 50] * numWord
topic_count = [0] * 50
doc_index = 0
# print(topic_seqs[0])
for topic_seq in topic_seqs:
    word_index = 0
    for topic in topic_seq:
        topic_count[topic] += 1
        # print(str(doc_index) + " " + str(word_index) + " " + str(topic) + " " + str(word_set.index(places_users[doc_index][word_index])))
        word_topic_count[word_set.index(places_users[doc_index][word_index])][topic] += 1
        word_index += 1
    doc_index += 1

print(word_topic_count[0])

# print(topic_word_count)

thresholds = {0.1}
# for i in range(1, 2):
#     thresholds.append(float(i) / 10)


class UserSimilarity:
    def __init__(self, index, sim):
        self.index = index
        self.sim = sim

    def __repr__(self):
        return str(self.index) + " " + str(self.sim)


per = 0.0
per1 = 0.0
for threshold in thresholds:
    for user_index1 in range(0, 100):
        uss = []
        diss = []
        dis_sum = 0.0
        under_threshold_count = 0
        places_similar_user = []
        user_index2 = 0
        for user in data.values:
            distance = np.linalg.norm(user - data.values[user_index1])
            dis_sum += distance
            diss.append(distance)
            if user_index2 != user_index1:
                uss.append(UserSimilarity(user_index2, distance))
            # if distance < threshold:
            #     if user_index2 != user_index1:
            #         under_threshold_count += 1
            #         places_similar_user.extend(places_users[user_index2])
            user_index2 += 1
        uss.sort(key=lambda x: x.sim)
        under_threshold_count = 200
        max_dis = uss[under_threshold_count-1].sim
        for us in uss[0: under_threshold_count]:
            places_similar_user.extend(places_users[us.index])
        # a = []
        # a.extend(places_users[user_index1])
        # a.extend(places_similar_user)
        # print(user_index1, end="\t")
        # print(round(max_dis, 2), end="\t")
        # print(under_threshold_count, end="\t")
        # print(len(set(places_similar_user)), end="\t")
        # print(len(set(places_users_test[user_index1])), end="\t")
        # print(len(set(places_similar_user) & set(places_users_test[user_index1])), end="\t")
        # print(round(float(len(set(places_similar_user) & set(places_users_test[user_index1]))) / len(set(places_users_test[user_index1])), 2), end="\t")
        # print(len(set(a) & set(places_users_test[user_index1])), end="\t")
        # per1 += round(float(len(set(a) & set(places_users_test[user_index1]))) / len(set(places_users_test[user_index1])), 2)
        # per += round(float(len(set(places_similar_user) & set(places_users_test[user_index1]))) / len(set(places_users_test[user_index1])), 2)
        # print(round(float(len(set(a) & set(places_users_test[user_index1]))) / len(set(places_users_test[user_index1])), 2), end="\t")
        # print(dis_sum / (len(data.values)-1), end="\t")
        # print("")

# print(per / 100)
# print(per1 / 100)
