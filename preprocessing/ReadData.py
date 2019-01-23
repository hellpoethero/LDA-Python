import pandas as pd


class ReadData:
    def __init__(self):
        self.words = []
        self.wordTopicCounts = []
        self.topicDocsDis = []
        self.topicSequences = []
        self.train = []
        self.test = []
        self.wordTopicDistribution = []
        self.tokenTopicSum = []
        self.wordDistribution = []
        self.locationList = None

    def read_topic_sequence(self, topic_sequences_path):
        with open(topic_sequences_path, "r") as inFile:
            for line in inFile:
                self.topicSequences.append(list(map(int, line.rstrip().split(" "))))
        return self.topicSequences

    def read_topic_doc(self, topic_docs_path):
        with open(topic_docs_path, "r") as inFile:
            for line in inFile:
                self.topicDocsDis.append(list(map(float, line.rstrip().split(" "))))
        return self.topicDocsDis

    def read_word_topic(self, word_topic_count_path):
        with open(word_topic_count_path, "r") as inFile:
            for line in inFile:
                self.words.append(int(line.rstrip().split(":")[0]))
                self.wordTopicCounts.append(list(map(int, line.split(":")[1].rstrip().split(" "))))
        tokenTopicSum = []
        for topicIndex in range(0, len(self.wordTopicCounts[0])):
            tokenTopicSum.append(0)
            for wordIndex in range(0, len(self.wordTopicCounts)):
                tokenTopicSum[topicIndex] += self.wordTopicCounts[wordIndex][topicIndex]
        for wordTopicCount in self.wordTopicCounts:
            wordDistribution = []
            for topicIndex in range(0, len(wordTopicCount)):
                wordDistribution.append(float(wordTopicCount[topicIndex]) / tokenTopicSum[topicIndex])
            self.wordTopicDistribution.append(wordDistribution)
        return self.wordTopicDistribution

    def read_train(self, train_path):
        self.train = []
        # train_path = "D:\Research\Dataset\checkin/user_checkin_above_10x10x5_us_train - Copy.txt"
        with open(train_path, "r") as inFile:
            for line in inFile:
                self.train.append(list(map(int, line.rstrip().split(" "))))
        return self.train

    def read_test(self, test_path):
        self.test = []
        # test_path = "D:\Research\Dataset\checkin/user_checkin_above_10x10x5_us_test - Copy.txt"
        with open(test_path, "r") as inFile:
            for line in inFile:
                self.test.append(list(map(int, line.rstrip().split(" "))))
        return self.test

    def read_location_list(self, file_path):
        self.locationList = pd.read_csv(file_path, sep=',', header=0)
        return self.locationList

    def get_users_places(self):
        doc_index = 0
        users_places = []
        for doc in self.train:
            for place in doc:
                users_places.append([doc_index, place])
            doc_index += 1
        df = pd.DataFrame(users_places)
        df.columns = ['user', 'place']
        return df
