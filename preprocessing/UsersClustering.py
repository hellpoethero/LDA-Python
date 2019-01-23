from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

inputFile = "D:/Research/Result/checkin/topic_doc_dis_1540973167751.csv"
data = pd.read_csv(inputFile, sep=' ', header=None)

# print(data)

cluster_n = 100
X = np.array(data.values)
kmeans = KMeans(n_clusters=cluster_n, random_state=0, n_init=10, max_iter=1000).fit(X)
print(kmeans.labels_)
# kmeans.predict([[0, 0], [4, 4]])
# print(kmeans.cluster_centers_)
counts = [0] * cluster_n
for i in kmeans.labels_:
    counts[i] += 1

print(counts)
