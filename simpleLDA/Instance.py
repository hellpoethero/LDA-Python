from simpleLDA import FeatureSequence


class Instance:
    target = []
    name = ''
    source = []
    properties = []
    locked = False

    def __init__(self):
        self.data = FeatureSequence.FeatureSequence()

    def set_sentence(self, tokens, dictionary):
        self.data.set_sentences(tokens, dictionary)

    def __repr__(self):
        return repr(self.data)
