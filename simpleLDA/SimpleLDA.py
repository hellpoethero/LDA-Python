import random

from simpleLDA import TopicAssignment, LabelSequence, SortedWord, SortedTopic


class SimpleLda:
    data = []  # the training instances and their topic assignments
    alphabet = []  # the alphabet for the input data
    topicAlphabet = []  # the alphabet for the topics
    numTopics = 1  # The number of topics requested
    numTypes = 1  # The size of the vocabulary
    alpha = 1  # Dirichlet(alpha,alpha,...) is the distribution over topics
    alphaSum = 1
    beta = 0.01  # Prior on per-topic multinomial distribution over words
    betaSum = 1
    oneDocTopicCounts = []  # indexed by <document index, topic index>
    typeTopicCounts = []  # indexed by <feature index, topic index>
    tokensPerTopic = []  # indexed by <topic index>
    showTopicsInterval = 50
    wordsPerTopic = 10

    def __init__(self, number_of_topics, alpha_sum, beta):
        self.data = []
        # self.topicAlphabet = topic_alphabet
        self.numTopics = number_of_topics

        self.alphaSum = alpha_sum
        self.alpha = alpha_sum / self.numTopics
        self.beta = beta
        self.random = random

        self.oneDocTopicCounts = [0] * self.numTopics
        self.tokensPerTopic = [0] * self.numTopics

    def add_instances(self, training):
        self.alphabet = training.dataAlphabet
        self.numTypes = len(self.alphabet)
        self.betaSum = self.beta * self.numTypes
        self.topicAlphabet = training.targetAlphabet

        for i in range(0, self.numTypes):
            temp_array = []
            for j in range(0, self.numTopics):
                temp_array.append(0)
            self.typeTopicCounts.append(temp_array)
        # print(self.typeTopicCounts)
        print(self.numTypes)

        doc = 0
        for instance in training.instances:
            doc += 1
            tokens = instance.data
            topics = []
            for i in range(0, tokens.length):
                topics.append(-1)
            topic_sequence = LabelSequence.LabelSequence(self.topicAlphabet, topics)

            topics = topic_sequence.features
            for position in range(0, tokens.length):
                topic = self.random.randint(0, self.numTopics - 1)
                topics[position] = topic
                self.tokensPerTopic[topic] += 1

                type1 = tokens.features[position]
                self.typeTopicCounts[type1][topic] += 1

            t = TopicAssignment.TopicAssignment(instance, topic_sequence)
            self.data.append(t)
            # print(str(doc) + " ", end=' ')
            # print(topic_sequence.features)
        # print(self.typeTopicCounts)
        # i = 0
        # for count in self.typeTopicCounts:
        #     print(self.alphabet[i], end='\t')
        #     print(count)
        #     i += 1

    def sample(self, iterations):
        for iteration in range(0, iterations):
            for doc in self.data:
                token_sequence = doc.instance.data
                topic_sequence = doc.topicSequence
                self.sample_topics_for_one_doc(token_sequence, topic_sequence)
        # print(self.tokensPerTopic)
        # sum = 0
        # for a in self.tokensPerTopic:
        #     sum += a
        # print(sum)
        # print(self.numTypes)
        self.top_word()
        # self.print_document_topics(10)

    def sample_topics_for_one_doc(self, token_sequence, topic_sequence):
        one_doc_topics = topic_sequence.features
        topic_weights_sum = 0.0
        doc_length = token_sequence.length
        local_topic_counts = [0] * self.numTopics

        for position in range(0, doc_length):
            local_topic_counts[one_doc_topics[position]] += 1

        topic_term_scores = [0.0] * self.numTopics

        for position in range(0, doc_length):
            type1 = token_sequence.features[position]
            old_topic = one_doc_topics[position]

            current_type_topic_counts = self.typeTopicCounts[type1]

            local_topic_counts[old_topic] -= 1
            self.tokensPerTopic[old_topic] -= 1
            current_type_topic_counts[old_topic] -= 1

            sum1 = 0.0
            for topic in range(0, self.numTopics):
                score = (self.alpha + local_topic_counts[topic]) * (
                        (self.beta + current_type_topic_counts[topic]) /
                        (self.betaSum + self.tokensPerTopic[topic]))
                sum1 += score
                topic_term_scores[topic] = score

            r = self.random.random()
            sample = r * sum1
            new_topic = -1
            while sample > 0.0:
                new_topic += 1
                sample -= topic_term_scores[new_topic]

            if new_topic == -1:
                break

            one_doc_topics[position] = new_topic
            local_topic_counts[new_topic] += 1
            self.tokensPerTopic[new_topic] += 1
            current_type_topic_counts[new_topic] += 1

    def top_word(self):
        for topic in range(0, self.numTopics):
            sorted_words = []
            for type1 in range(0, self.numTypes):
                sorted_words.append(SortedWord.SortedWord(
                    type1, self.alphabet[type1], self.typeTopicCounts[type1][topic]))
                # print(type1)
                # print(self.typeTopicCounts[type1][topic])
            sorted_words = sorted(sorted_words, key=lambda word: word.count, reverse=True)
            for i in range(0, self.wordsPerTopic):
                print(sorted_words[i].word, end=' ')
            print()

    def print_document_topics(self, max_topic):
        if max_topic < 0 or max_topic > self.numTopics:
            max_topic = self.numTopics

        for doc in self.data:
            topic_counts = [0] * self.numTopics
            sorted_topics = []
            topic_sequence = doc.topicSequence
            current_doc_topics = topic_sequence.features
            # print(doc.instance.data)

            doc_len = len(current_doc_topics)

            for token in range(0, doc_len):
                topic_counts[current_doc_topics[token]] += 1

            for topic in range(0, self.numTopics):
                sorted_topics.append(SortedTopic.SortedTopic(
                    topic, float(topic_counts[topic]) / float(doc_len)))

            sorted_topics = sorted(sorted_topics, key=lambda topic1: topic1.weight, reverse=True)

            for i in range(0, max_topic):
                print(sorted_topics[i], end=" ")
            print()
