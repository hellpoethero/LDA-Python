import numpy as np
import matplotlib.pyplot as plt

sumA = 0.0
a = []
inputFile = "D:/Research/Result/checkin/pos.txt"
with open(inputFile, "r") as lines:
    for line in lines:
        pos = int(float(line))
        a.append(pos / 1)
        sumA += pos

    print(sumA / len(a))

# print(poss)
a.sort()
print(len(a))
print(a[0])
print(a[1])
print(a[-1])
print(a.count(0))
# a_set = set(a)
# xxx = []
# for value in a_set:
#     xxx.append(a.count(value))

# print(xxx)

# the histogram of the data
n, bins, patches = plt.hist(a, 2000, density=False, facecolor='g', alpha=0.75)

plt.xlabel('Average position')
plt.ylabel('Number of user')
plt.title('Histogram of count of user by average position')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([0, 10000, 0, 700])
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

