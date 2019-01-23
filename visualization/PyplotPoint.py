import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim

us_ca = pd.read_csv("D:/Research/Dataset/checkin/us_canada.txt", sep=',', header=0)
lats = us_ca['lat'].values.tolist()
lons = us_ca['lon'].values.tolist()
print(lats[0:10])
print(lons[0:10])
# plt.plot(lons, lats, 'ro')
# plt.axis([-160, -60, 20, 70])
# plt.show()

us_map = Basemap(llcrnrlon=-120, llcrnrlat=20, urcrnrlon=-40, urcrnrlat=60,
        projection='lcc', lat_1=32, lat_2=45, lon_0=-90)

# load the shapefile, use the name 'states'
us_map.readshapefile('st99_d00', name='states', drawbounds=True)

us_map.drawcoastlines()
# Get the location of each city and plot it
geolocator = Nominatim()
x, y = us_map(lons, lats)
us_map.plot(x, y, 'ro')
plt.axis('on')
plt.show()
