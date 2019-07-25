import ReadData
import Lda
import Lda01
import Lda02
import time
import os

path = "D:/Download/data/"
datasetName = "tw_ny/"
# go_sf go_ny tw_ny tw_oc lastfm reddit_sample

df = ReadData.read(path, datasetName)

Ks = [3, 4, 5, 7, 10]
iters = [100]
top_k = 100

folder_name = "D:/Research/Project/LDA/results/" + datasetName[:-1] + "_" + str(int(time.time()))
os.makedirs(folder_name)
#
# with open(folder_name + "/" + datasetName[:-1] + ".csv", "w") as outFile:
# 	Lda01.init(df, Ks, iters, top_k, datasetName, outFile, folder_name)

with open(folder_name + "/" + datasetName[:-1] + ".csv", "w") as outFile:
	Lda02.init(df, Ks, iters, top_k, datasetName, outFile, folder_name)
# datasetName = "lastfm/"
# folder_name = "D:/Research/Project/LDA/results/" + datasetName[:-1] + "_" + str(time.time())
# os.makedirs(folder_name)
#
# with open(folder_name + "/" + datasetName[:-1] + ".csv", "w") as outFile:
# 	Lda01.init(df, [], [], top_k, datasetName, outFile, folder_name)
