import ReadData
import SortedWord
import numpy as np
import matplotlib.pyplot as plt

data = ReadData.ReadData()

trainPath = "D:\Research\Dataset\checkin/user_checkin_above_10x10x5_us_train - Copy.txt"
train = data.read_train(trainPath)

a = []
for doc in train:
    docLen = len(doc)
    docWordsLen = len(set(doc))
    avg_repeat = docLen / docWordsLen
    a.append(avg_repeat)

print(np.average(a))
print(np.max(a))
print(a.index(np.max(a)))
n, bins, patches = plt.hist(a, 100, density=False, facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()
