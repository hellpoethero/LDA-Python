from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

inputFile = "D:/Research/Dataset/checkin/Gowalla_places.txt"
data = pd.read_csv(inputFile, sep=',', header=None)

print(len(data))
coordinate_values = data[data[3] >= 10].iloc[:, 1:3].values
id_values = data[data[3] >= 10].iloc[:, 0:1].values
print(coordinate_values)
# print(id_value)
print(len(coordinate_values))
# print(len(id_value))

cluster_n = 11
X = np.array(coordinate_values)
kmeans = KMeans(n_clusters=cluster_n, random_state=0, n_init=10, max_iter=1000).fit(X)
print(kmeans.labels_)
# kmeans.predict([[0, 0], [4, 4]])
print(kmeans.cluster_centers_)
counts = [0] * cluster_n
for i in kmeans.labels_:
    counts[i] += 1

print(counts)

data_with_label = []
for i in range(0, cluster_n):
    data_with_label.append([])
i = 0
for label in kmeans.labels_:
    data_with_label[label].append(
        [id_values[i][0], coordinate_values[i][0], coordinate_values[i][1]])
    i += 1

american_places = []
american_places.extend(data_with_label[1])
american_places.extend(data_with_label[4])
american_places.extend(data_with_label[5])
print(len(american_places))
a = list(filter(lambda x: x[1] * -3.03688 - 18.3269 <= x[2] and x[1] > 25.15, american_places))

print(len(a))
sort_american_places = []
sort_american_places.append(sorted(a, key=lambda x: x[1]))
sort_american_places.append(sorted(a, key=lambda x: x[2]))
print(sort_american_places[0][0])
print(sort_american_places[0][-1])
print(sort_american_places[1][0])
print(sort_american_places[1][-1])

# outputData = pd.DataFrame(a)
# outputData.to_csv("D:/Research/Dataset/checkin/us_canada.txt")
# with open("D:/Research/Dataset/checkin/us_canada.txt", "w") as outFile:

# sort_data = []
# sort_data.append(sorted(data_with_label[4], key=lambda x: x[0]))
# sort_data.append(sorted(data_with_label[4], key=lambda x: x[1]))
# # print(data_with_label[7])
# print(sort_data[0][0])
# print(sort_data[0][-1])
# print(sort_data[1][0])
# print(sort_data[1][-1])

