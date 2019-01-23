import matplotlib.pyplot as plt

alphas = []
with open("D:\Research\Result/alphas.txt", "r") as inFile:
    for line in inFile:
        alphas.append(float(line.rstrip()))


n, bins, patches = plt.hist(alphas, 500, density=True, facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()
