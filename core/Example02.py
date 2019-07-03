import ReadData
import Lda
import Lda01
import time

path = "D:/Download/data/"
datasetName = "go_ny/"
# go_sf go_ny tw_ny tw_oc lastfm reddit_sample

df = ReadData.read(path, datasetName)

Ks = [3, 4, 5, 7, 10]
iters = [10, 100]
top_k = 100

with open("D:/Research/Project/LDA/results/" + datasetName[:-1] + "_" + str(time.time()) + ".csv", "w") as outFile:
	Lda01.init(df, Ks, iters, top_k, datasetName, outFile)

