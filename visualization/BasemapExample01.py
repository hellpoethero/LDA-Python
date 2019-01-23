from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

map1 = Basemap(projection='cyl')

map1.drawmapboundary(fill_color='aqua')
map1.fillcontinents(color='coral',lake_color='aqua')
map1.drawcoastlines()

plt.show()
