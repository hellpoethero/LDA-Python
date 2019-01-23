# use k-means to cluster places
# visualize clusters

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

inputFile = "D:/Research/Dataset/checkin/us_canada.txt"
data = pd.read_csv(inputFile, sep=',', header=0)

print(len(data))

coordinate_values = data.iloc[:, 2:4].values
id_values = data.iloc[:, 0:1].values
# print(coordinate_values)

cluster_n = 500
# X = np.array(coordinate_values)
# kmeans = KMeans(n_clusters=cluster_n, random_state=0, n_init=10, max_iter=1000).fit(X)
# # print(kmeans.labels_)
# # print(kmeans.cluster_centers_)
# counts = [0] * cluster_n
# for i in kmeans.labels_:
#     counts[i] += 1

# mms = MinMaxScaler()
# mms.fit(coordinate_values)
# data_transformed = mms.transform(coordinate_values)

Sum_of_squared_distances = []
K = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
# K = [50, 100]
for k in K:
    km = KMeans(n_clusters=k, random_state=0, n_init=10, max_iter=1000)
    km = km.fit(coordinate_values)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# print(counts)

# visualize data

# cluster_coordinates = np.array(kmeans.cluster_centers_)
# lons = cluster_coordinates[:, 1]
# lats = cluster_coordinates[:, 0]
# # print(len(lons))
# # print(len(lats))
#
# us_map = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
#         projection='lcc', lat_1=32, lat_2=45, lon_0=-95)
#
# # load the shapefile, use the name 'states'
# us_map.readshapefile('st99_d00', name='states', drawbounds=True)
#
# # Get the location of each city and plot it
# # geolocator = Nominatim()
# x, y = us_map(lons, lats)
# us_map.plot(x, y, 'go')
# plt.show()
