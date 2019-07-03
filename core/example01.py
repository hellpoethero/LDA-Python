import ReadData
import Lda

path = "D:/Download/data/"
# datasetName = "lastfm_sample_190617/"
datasetName = "go_sf/"
# "/Users/yoonsik/Downloads/

df = ReadData.read(path, datasetName)
K = 5
iter = 10
top_k = 100
Lda.run(df, K, iter, top_k)

