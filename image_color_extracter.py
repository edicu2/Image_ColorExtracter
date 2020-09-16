import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

# start_time
start = time.time() 

# Reduce image to reduce clustering time
# desize 영역 보간법( INTER_LINEAR ) 사용
src = cv2.imread("test.png")
image = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
print(image.shape) # (92, 272, 3) => height, width, channel 

# BGR -> RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# height, width 통합
image = image.reshape((image.shape[0] * image.shape[1], 3))
print(image.shape) # (10040, 3) => area, channel 

# clustering
k = 3
clt = KMeans(n_clusters = k)
clt.fit(image)

# 각 category center RGB
for center in clt.cluster_centers_:
    intList = list(map(int, center))
    print(intList)

# 각 category proportion
def proportion_histogram(clt):
    # histogram(도수분포표)
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # 비율 구하기 (각 hist 갯수 /전체 갯수 )
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram proportion  
    return hist

hist = proportion_histogram(clt)
print(hist) # ex) category 5 => [ 0.68881873  0.09307065  0.14797794  0.04675512  0.02337756 ]


# barChart setting
def plot_colors(hist, centroids):
    # 
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

# show our color bart
bar = plot_colors(hist, clt.cluster_centers_)
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()


# end time check 
print("time :", time.time() - start)
