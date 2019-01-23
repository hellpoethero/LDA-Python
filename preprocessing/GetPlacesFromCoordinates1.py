import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim

us_ca = pd.read_csv("D:/Research/Dataset/checkin/us_canada.txt", sep=',', header=0)
places_in_city = us_ca[(us_ca['lat'] > 40.495006) & (us_ca['lat'] < 40.915483) &
           (us_ca['lon'] > -74.011864) & (us_ca['lon'] < -73.701145)]
lats = places_in_city['lat'].values.tolist()
lons = places_in_city['lon'].values.tolist()
ids = places_in_city['id'].values.tolist()
print(len(ids))

places_in_city.to_csv('D:/Research/Dataset/checkin/newyork.txt', index=False)

inputFile = "D:\Research\Dataset\checkin/Gowalla_totalCheckins_chekin10.txt"
data = pd.read_csv(inputFile, sep='\t', header=None)
checkins_in_city = data[data[1].isin(places_in_city['id'])]
x1 = checkins_in_city.groupby([0])[1]
x1_list = x1.apply(list)

indexes = x1_list.index.tolist()
place_ids = x1_list[indexes[8]]
coordinates = places_in_city[places_in_city['id'].isin(ids)]
# plt.grid(True)
# plt.axis([-160, -60, 20, 70])
# plt.plot(lons, lats, 'ro')
# plt.show()

us_map = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=32, lat_2=45, lon_0=-95)

# load the shapefile, use the name 'states'
us_map.readshapefile('st99_d00', name='states', drawbounds=True)

# Get the location of each city and plot it
# geolocator = Nominatim()
x, y = us_map(lons, lats)
us_map.plot(x, y, 'ro')
plt.show()
