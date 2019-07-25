import ReadData01
import Lda03
import time
import os

path = "D:/Download/data/"
datasetName = "lastfm/"
# go_sf go_ny tw_ny tw_oc lastfm reddit_sample

folder_name = "D:/Research/Project/LDA/results/190718_2/" + datasetName[:-1]
# if !os.direxists(folder_name):
os.makedirs(folder_name)

df = ReadData01.read01(path, datasetName)

Ks = [4]
iters = [5]
top_k = 100

with open(folder_name + "/" + datasetName[:-1] + ".csv", "w") as outFile:
	Lda03.init(df, Ks, iters, top_k, datasetName, outFile, folder_name)
# datasetName = "lastfm/"
# folder_name = "D:/Research/Project/LDA/results/" + datasetName[:-1] + "_" + str(time.time())
# os.makedirs(folder_name)
#
# with open(folder_name + "/" + datasetName[:-1] + ".csv", "w") as outFile:
# 	Lda01.init(df, [], [], top_k, datasetName, outFile, folder_name)
