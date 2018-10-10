import os
from simpleLDA import PreProcessing
from simpleLDA import Instance


class InstanceList:
    def __init__(self):
        self.dataAlphabet = []
        self.targetAlphabet = []
        self.instances = []
        self.pre_processing = PreProcessing.PreProcessing()

    def set_pre_processing(self, pre_processing):
        self.pre_processing = pre_processing

    def load_directory(self, directory):
        for file in os.listdir(directory):
            filename = directory + "\\" + file
            self.load_file(filename)

    def load_file(self, file):
        # print(file)
        with open(file, 'r') as inputFile:
            lines = []
            for line in inputFile:
                lines.append(line)
            for line in lines:
                # print(line)
                tokens = self.pre_processing.process(line)
                # print(tokens)
                temp = []
                for token in tokens:
                    if token not in self.dataAlphabet:
                        self.dataAlphabet.append(token)
                        temp.append(len(self.dataAlphabet) - 1)
                    else:
                        index = self.dataAlphabet.index(token)
                        temp.append(index)

                instance = Instance.Instance()
                instance.set_sentence(temp, self.dataAlphabet)
                self.instances.append(instance)
                # print(instance.data.features)
        # print(self.dataAlphabet)
