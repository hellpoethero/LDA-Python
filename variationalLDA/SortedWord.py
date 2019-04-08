class SortedWord:
    def __init__(self, index, word, count):
        self.index = index
        self.word = word
        self.count = count

    def __repr__(self):
        return repr((self.index, self.word, self.count))
