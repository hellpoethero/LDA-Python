from simpleLDA import InstanceList, PreProcessing, SimpleLDA
import time
import datetime

a = InstanceList.InstanceList()
pre_processing = PreProcessing.PreProcessing()
pre_processing.set_stop_words("D:\Research\Project\LDA\stoplists\en.txt")
a.set_pre_processing(pre_processing)
# a.load_directory('D:\Research\Project\mallet\sample-data\web\en')

print("Load file")

start_read_time = time.time()
a.load_file('D:/Download/ap/Associated Press.txt')
end_read_time = time.time()
print("Read file duration: " + str(end_read_time - start_read_time))

lda = SimpleLDA.SimpleLda(10, 50, 0.01)
print("Run sampling")
start_time = time.time()
lda.add_instances(a)
end_time = time.time()
print(
    "start: "
    + datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    + " end: "
    + datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    + " duration: " + str(end_time - start_time))

# lda.sample(1)
