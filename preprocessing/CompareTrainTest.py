import matplotlib.pyplot as plt


trainFileStr = "D:/Research/Dataset/checkin/user_checkin_above_10x10x5_us_train - Copy.txt"
testFileStr = "D:/Research/Dataset/checkin/user_checkin_above_10x10x5_us_test - Copy.txt"

counts = []
trains = []
trainSets = []
docTrainLens = []
tests = []
testSets = []
docTestLens = []
a = []
with open(trainFileStr, "r") as trainFile:
    for line in trainFile:
        doc = list(map(int, line.rstrip().split(" ")))
        docTrainLens.append(len(doc))
        trains.append(doc)
        places = set(doc)
        trainSets.append(places)
        for place in places:
            a.append(doc.count(place))
with open(testFileStr, "r") as testFile:
    for line in testFile:
        doc = list(map(int, line.rstrip().split(" ")))
        docTestLens.append(len(doc))
        places = set(doc)
        testSets.append(places)
        tests.append(doc)

full_match_count = 0
match_places_len = 0
test_len = 0
for i in range(0, len(trainSets)):
    oneTimePlaces = []
    for word in trainSets[i]:
        if trains[i].count(word) == 1:
            oneTimePlaces.append(word)
    # match_places = set(oneTimePlaces) & testSets[i]
    match_places = trainSets[i] & testSets[i]
    match_places_len += len(match_places)
    test_len += len(testSets[i])
    if len(match_places) == len(testSets[i]):
        full_match_count += 1
    counts.append(len(match_places) / len(testSets[i]))
    print(docTrainLens[i], end="\t")
    print(len(trainSets[i]), end="\t")
    print(len(oneTimePlaces), end="\t")
    print(docTestLens[i], end="\t")
    print(len(testSets[i]), end="\t")
    print(len(match_places), end="\t")
    print(len(set(oneTimePlaces) & testSets[i]))
# print(full_match_count)
# print(match_places_len)
# print(test_len)

# print(a)
n, bins, patches = plt.hist(a, 100, density=False, facecolor='g', alpha=0.75)

# plt.axis([0, 1, 0, 10000])
plt.grid(True)
# plt.show()
