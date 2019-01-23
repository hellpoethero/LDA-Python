import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim

us_ca = pd.read_csv("D:/Research/Dataset/checkin/us_canada.txt", sep=',', header=0)
# lats = us_ca['lat'].values.tolist()
# lons = us_ca['lon'].values.tolist()
# print(lats[0:10])
# print(lons[0:10])
sf = us_ca[(us_ca['lat'] > 37.706516) & (us_ca['lat'] < 37.833998) &
           (us_ca['lon'] > -122.531080) & (us_ca['lon'] < -122.350436)]
lats = sf['lat'].values.tolist()
lons = sf['lon'].values.tolist()
ids = sf['id'].values.tolist()
# print(ids[0:10])

sf.to_csv('D:/Research/Result/checkin/sanfrancisco.txt', index=False)


inputFile = "D:\Research\Dataset\checkin/Gowalla_totalCheckins_chekin10.txt"
data = pd.read_csv(inputFile, sep='\t', header=None)
sf_checkins = data[data[1].isin(sf['id'])]
# sf_checkins = data
x1 = sf_checkins.groupby([0])[1]
x1_list = x1.apply(list)
# print(x1_list.tolist())
# print(x1.count())

# sf_user_checkins = x1_list.tolist()
# for user in sf_user_checkins:
#         if len(user) > 10:
#                 print("a b " + " ".join(list(map(str, user))))
# print(len(sf_user_checkins))

indexes = x1_list.index.tolist()
place_ids = x1_list[indexes[8]]
# print(place_ids)
coordinates = sf[sf['id'].isin(ids)]
# coordinates = us_ca[us_ca['id'].isin(place_ids)]
# print(coordinates)
# lats = coordinates['lat'].values.tolist()
# lons = coordinates['lon'].values.tolist()
# print(lats[0:10])
# print(lons[0:10])
plt.grid(True)
plt.axis([-160, -60, 20, 70])
plt.plot(lons, lats, 'ro')
# plt.show()

us_map = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=32, lat_2=45, lon_0=-95)

# load the shapefile, use the name 'states'
us_map.readshapefile('st99_d00', name='states', drawbounds=True)

# Get the location of each city and plot it
# geolocator = Nominatim()
x, y = us_map(lons, lats)
us_map.plot(x, y, 'ro')
# plt.show()
