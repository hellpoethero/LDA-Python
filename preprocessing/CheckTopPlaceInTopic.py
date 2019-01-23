import ReadData
import SortedWord
import numpy as np
import matplotlib.pyplot as plt

data = ReadData.ReadData()

baseDir = "D:/Research/Result/checkin/"
wordTopicCountPath = baseDir+"word_topic_count_1542169953208.txt"
topicDocsPath = baseDir+"topic_doc_dis_1542169953208.csv"
topicSequencesPath = baseDir+"topic_seq_1542169953208.csv"
locationListPath = "D:/Research/Dataset/checkin/us_canada.txt"

wordTopicDistribution = data.read_word_topic(wordTopicCountPath)

words = data.words
topicDocsDis = data.read_topic_doc(topicDocsPath)
topicSequences = data.read_topic_sequence(topicSequencesPath)
locationList = data.read_location_list(locationListPath)

testPath = "D:\Research\Dataset\checkin/user_checkin_above_10x10x5_us_test - Copy.txt"
test = data.read_test(testPath)

trainPath = "D:\Research\Dataset\checkin/user_checkin_above_10x10x5_us_train - Copy.txt"
train = data.read_train(trainPath)

predictedWords = []
for i in range(0, len(wordTopicDistribution)):
    predictedWords.append(SortedWord.SortedWord(i, words[i], 0))


docIndex = 23893
for topics in topicDocsDis[docIndex:docIndex+1]:
    topicIndex = 0
    for topic in topics:
        if topic > 0:
            # print(topicIndex, end=" ")
            wordIndex = 0
            for word in wordTopicDistribution:
                if word[topicIndex] > 0:
                    predictedWords[wordIndex].count += word[topicIndex] * topic
                wordIndex += 1
        topicIndex += 1
    print()
    nonZeroWords = []
    for word in predictedWords:
        if word.count > 0:
            nonZeroWords.append(word)
    nonZeroWords = sorted(nonZeroWords, key=lambda word: word.count, reverse=True)
    # print(nonZeroWords[0:50])
    sortedWords = []
    for word in nonZeroWords:
        sortedWords.append(word.word)
    testWords = list(set(test[docIndex]))
    trainWords = list(set(train[docIndex]))
    duplicatedWords = list(set(testWords) & set(trainWords))
    # for word in trainWords:
    #     print(word, end="\t")
    #     print(train[docIndex].count(word))
    # print(len(trainWords))
    # print(len(train[docIndex]))
    for word in testWords:
        if word in sortedWords:
            print(sortedWords.index(word), end="\t")
            print(word)
        else:
            print("", end="\t")
            print(word, end="\t")
            # topicIndex = 0
            # for topic in wordTopicDistribution[words.index(int(word))]:
            #     if topic > 0:
            #         print(topicIndex, end="\t")
            #     topicIndex += 1
            print()
    # for word in trainWords:
    #     if word not in duplicatedWords:
    #         print(sortedWords.index(word), end="\t")
    #         print(word, end="\t")
    #         print(train[docIndex].count(int(word)))
    #         pass
    #     else:
    #         print(sortedWords.index(word), end="\t")
    #         print(word, end="\t")
    #         print(train[docIndex].count(int(word)), end="\t")
    #         print("duplicated")

    # wordIndex = 0
    # for word in sortedWords:
    #     if word not in testWords:
    #         print(wordIndex, end="\t")
    #         print(word, end="\t")
    #         print(nonZeroWords[wordIndex])
    #     wordIndex += 1
    #     if wordIndex > 1000:
    #         break

    docIndex += 1
    for i in range(0, len(wordTopicDistribution)):
        predictedWords[i].count = 0
    break

# users_places = data.get_users_places()
# unique_user_place_counts = users_places.groupby(['user', 'place']).size().groupby(['place']).size().values.tolist()
# place_counts = users_places.groupby(['place'])['user'].size().values.tolist()
# places = users_places.groupby(['place']).apply(list).index.tolist()
# print(len(places))
#
# a = []
# for placeIndex in range(0, len(place_counts)):
#     avg_value = float(place_counts[placeIndex]) / unique_user_place_counts[placeIndex]
#     a.append(SortedWord.SortedWord(words.index(places[placeIndex]), places[placeIndex], avg_value))
# a = sorted(a, key=lambda word: word.count, reverse=True)
# print(a[0:100])

# n, bins, patches = plt.hist(a, 100, density=False, facecolor='g', alpha=0.75)
# plt.grid(True)
# plt.show()
