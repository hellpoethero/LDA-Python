class FeatureSequence:
    dictionary = []
    features = []
    length = 0

    def __init__(self):
        pass

    def set_sentences(self, tokens, dictionary):
        self.dictionary = dictionary
        # temp_features = []
        # for token in tokens:
        #     temp_features.append(dictionary.index(token))
        # self.features = temp_features
        self.features = tokens
        self.length = len(self.features)

    def __repr__(self):
        sentence = ""
        for feature in self.features:
            sentence += self.dictionary[feature] + " "
        return sentence
