import re


class PreProcessing:
    def __init__(self):
        self.stop_words = []
        self.special_characters = True
        self.to_lower = True
        self.must_contain_word = True

    def process(self, string):
        if self.special_characters:
            string = re.sub("[^0-9a-zA-Z ]+", '', string)
        if self.to_lower:
            string = string.lower()
        tokens = string.split()
        if self.must_contain_word:
            temp_result = []
            for token in tokens:
                if re.match("[a-z]+", token) is not None:
                    # print(token)
                    temp_result.append(token)
            tokens = temp_result
        if len(self.stop_words) > 0:
            temp_result = []
            for token in tokens:
                if token not in self.stop_words:
                    temp_result.append(token)
            tokens = temp_result
        return tokens

    def set_to_lower(self, to_lower):
        self.to_lower = to_lower

    def set_must_contain_word(self, must_contain_word):
        self.must_contain_word = must_contain_word

    def check_stop_words(self, token):
        if token not in self.stop_words:
            return token
        else:
            return None

    def remove_stop_words(self, tokens):
        result = []
        for token in tokens:
            if token not in self.stop_words:
                result.append(token)
        return result

    def set_stop_words(self, file):
        with open(file, 'r') as inputFile:
            for line in inputFile:
                # print(line.rstrip())
                self.stop_words.append(line.rstrip())
            # print(self.stop_words)

    def set_special_characters(self, string):
        self.special_characters = string
