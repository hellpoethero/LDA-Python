# Draw the locations of cities on a map of the US

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim
import math

cities = [["Chicago", 10],
          ["Boston", 10],
          ["New York", 5],
          ["San Francisco", 25]]
scale = 5

map = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=32, lat_2=45, lon_0=-95)

# load the shapefile, use the name 'states'
map.readshapefile('st99_d00', name='states', drawbounds=True)

# Get the location of each city and plot it
geolocator = Nominatim()
# for (city, count) in cities:
#     loc = geolocator.geocode(city)
#     x, y = map(loc.longitude, loc.latitude)
#     map.plot(x, y, marker='o', color='Red', markersize=int(math.sqrt(count))*scale)
plt.show()
