class SortedTopic:
    def __init__(self, index, weight):
        self.index = index
        self.weight = weight

    def __repr__(self):
        return repr((self.index, self.weight))
