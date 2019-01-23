import numpy as np
import matplotlib.pyplot as plt
import ReadData
import SortedWord


data = ReadData.ReadData()

baseDir = "D:/Research/Result/checkin/"
wordTopicCountPath = baseDir+"word_topic_count_1544521701955.txt"
topicDocsPath = baseDir+"topic_doc_dis_1544521701955.csv"
topicSequencesPath = baseDir+"topic_seq_1544521701955.csv"

wordTopicDistribution = data.read_word_topic(wordTopicCountPath)

words = data.words
topicDocsDis = data.read_topic_doc(topicDocsPath)
topicSequences = data.read_topic_sequence(topicSequencesPath)

trainPath = "D:\Research\Dataset\checkin/user_checkin_above_10x10x5_us_train - Copy.txt"
train = data.read_train(trainPath)

testPath = "D:\Research\Dataset\checkin/user_checkin_above_10x10x5_us_test - Copy.txt"
test = data.read_test(testPath)

unique_user_place_counts = data.get_users_places().groupby(['user', 'place']).size().groupby(['place']).size()
place_list = unique_user_place_counts.index.tolist()
place_count_list = unique_user_place_counts.values.tolist()
place_frequencies = []
for word in words:
    wordIndex = place_list.index(word)
    place_frequencies.append(place_count_list[wordIndex])

tokenCount = 0
rankSum = 0.0
averageRanks = []
docIndex = 0
for doc in train:
    train_word_set = list(set(doc))
    wordByTopicProbabilities = []

    topicDisValues = []
    topicDisIndexes = []
    topicIndex = 0
    for topicDisOfDoc in topicDocsDis[docIndex]:
        if topicDisOfDoc > 0:
            topicDisValues.append(topicDisOfDoc)
            topicDisIndexes.append(topicIndex)
        topicIndex += 1

    # print(topicDocsDis)
    # print(topicDisIndexes)
    # print(topicDisValues)

    for word in words:
        wordIndex = words.index(word)
        wordDistribution = wordTopicDistribution[wordIndex]
        wordByTopicSum = 0.0
        topicIndex = 0
        for topicDisOfDoc in topicDisValues:
            if wordDistribution[topicDisIndexes[topicIndex]] > 0:
                wordByTopicSum += topicDisOfDoc * wordDistribution[topicDisIndexes[topicIndex]]
            topicIndex += 1
        # wordByTopicSum *= place_frequencies[wordIndex]
        wordByTopicProbabilities.append(SortedWord.SortedWord(wordIndex, word, wordByTopicSum))
    # print(wordByDocProbabilities)
    # print(wordByTopicProbabilities)
    sortedWordByTopic = sorted(wordByTopicProbabilities, key=lambda word_dis: word_dis.count, reverse=True)

    predictedWord = []
    for sortedWord in sortedWordByTopic:
        if sortedWord.count == 0:
            break
        predictedWord.append(sortedWord.word)

    ranks = []
    # print(len(set(test[docIndex])))
    # print(len(predictedWord))
    # tokenCount += len(set(test[docIndex]))
    for word in set(test[docIndex]):
        if word in predictedWord:
            rank = predictedWord.index(word)
            rankSum += rank
            tokenCount += 1
            # print(str(word) + ":" + str(rank), end=" ")
            # ranks.append(predictedWord.index(word))
            # print(predictedWord.index(word))
    # print()
    print(rankSum)

    # if len(ranks) > 0:
    #     avg_rank = np.average(ranks)
    #     averageRanks.append(avg_rank)
    #     print(avg_rank)

    # print(sorted_wordByTopic)

    # print(docIndex + 1, end="\t")
    # print(len(doc), end="\t")
    # print(len(train_word_set), end="\t")
    # average probability of word in a doc
    # higher value means that few words, high probability
    # lower value means that many words, low probability
    # print(round(wordByDocProbabilitiesAvg, 6), end="\t")

    # print(len(set(train_word_set) & set(test[docIndex])))

    docIndex += 1
    if docIndex > 100:
        break

print("---------------")
# print(np.average(averageRanks))
print(rankSum / tokenCount)
