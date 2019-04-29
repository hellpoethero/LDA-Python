from LdaEstimate import LdaEstimate
import time


start = time.time()
LdaEstimate.estimate(0.1, 100, "settings.txt", "D:/Research/Project/lda-c-master/example/ap/ap.dat",
					 "D:/Research/Project/lda-c-master/example/ap/vocab.txt", "seeded", "")
end = time.time()
print(end - start)
