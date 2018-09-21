import os
import re
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
            for line in inputFile:
                # print(line)
                tokens = self.pre_processing.process(line)
                # print(tokens)
                for token in tokens:
                    if token not in self.dataAlphabet:
                        self.dataAlphabet.append(token)
                instance = Instance.Instance()
                instance.set_sentence(tokens, self.dataAlphabet)
                self.instances.append(instance)
                # print(instance.data.features)
        # print(self.dataAlphabet)
