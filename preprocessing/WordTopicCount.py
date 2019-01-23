import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Nominatim


class SortedWord:
    def __init__(self, index, word, count):
        self.index = index
        self.word = word
        self.count = count

    def __repr__(self):
        return repr((self.index, self.word, self.count))


words = []
baseDir = "D:/Research/Result/checkin/"
wordTopicCounts = []
wordTopicCountPath = baseDir+"word_topic_count_1542169953208.txt"

wordTopicDistribution = []
with open(wordTopicCountPath, "r") as inFile:
    for line in inFile:
        words.append(int(line.rstrip().split(":")[0]))
        wordTopicCounts.append(list(map(int, line.split(":")[1].rstrip().split(" "))))
# print(len(words))
# print(len(wordTopicCounts))
tokenTopicSum = []
for topicIndex in range(0, len(wordTopicCounts[0])):
    tokenTopicSum.append(0)
    for wordIndex in range(0, len(wordTopicCounts)):
        tokenTopicSum[topicIndex] += wordTopicCounts[wordIndex][topicIndex]
# print(tokenTopicSum)
for wordTopicCount in wordTopicCounts:
    wordDistribution = []
    for topicIndex in range(0, len(wordTopicCount)):
        wordDistribution.append(float(wordTopicCount[topicIndex]) / tokenTopicSum[topicIndex])
    wordTopicDistribution.append(wordDistribution)
# print(wordTopicDistribution[0])

us_ca = pd.read_csv("D:/Research/Dataset/checkin/us_canada.txt", sep=',', header=0)
for topic in range(0, 50):
    # topic = 3
    temp = []
    tempPlaces = []
    i = 0
    for word in wordTopicCounts:
        if word[topic] > 100:
            temp.append(SortedWord(i, words[i], word[topic]))
            tempPlaces.append(words[i])
        i += 1

    sorted_words = sorted(temp, key=lambda word: word.count, reverse=True)
    print(sorted_words[0:10])

    coordinates = us_ca[us_ca['id'].isin(tempPlaces)]
    lats = coordinates['lat'].values.tolist()
    lons = coordinates['lon'].values.tolist()

    plt.figure(topic)
    us_map = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
            projection='lcc', lat_1=32, lat_2=45, lon_0=-95)

    # load the shapefile, use the name 'states'
    us_map.readshapefile('st99_d00', name='states', drawbounds=True)

    # Get the location of each city and plot it
    geolocator = Nominatim()
    x, y = us_map(lons, lats)
    us_map.plot(x, y, 'ro')
    plt.savefig("fig/fig_"+str(topic)+".jpg")
    plt.close()
    # plt.show()
