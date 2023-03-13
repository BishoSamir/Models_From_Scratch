import numpy as np
from data import getData
import matplotlib.pyplot as plt
import pandas as pd 
class Kmeans:

    def __init__(self, k = 3 , nIter = 100):
        self.k = k
        self.nIter = nIter
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []
        self.choosedClusters = []
        self.choosedCentroids = []
        self.minVariance = 1e9

    def predict(self,  X):
        self.X = X 
        self.observations , self.features = X.shape
        for _ in range(self.nIter):
            idxCentroids = np.random.choice(self.observations , self.k , replace=False)
            self.centroids = np.array([X[i] for i in idxCentroids])
            oldCentroids = np.array([centroid + 1 for centroid in self.centroids])
            
            while(not(oldCentroids == self.centroids).all()):
                self.clusters = self.getClusters()
                #Update Centroids
                oldCentroids = self.centroids
                self.centroids = self.updateCentroids()

            totalVariance = self.getVariance()
            if(totalVariance < self.minVariance):
                self.choosedCentroids = self.centroids
                self.choosedClusters = self.clusters
                self.minVariance = totalVariance
        
        self.centroids = self.choosedCentroids
        self.clusters = self.choosedClusters
        return self.choosedClusters , self.choosedCentroids
                
    def getVariance(self):
        centroids = self.centroids
        clusters = self.clusters
        variance = []
        for idx , cluster in enumerate(clusters):
            varWithinCluster = 0
            for i in cluster:
                dist = np.sum((self.X[i]-centroids[idx])**2)
                varWithinCluster += dist 
            variance.append(varWithinCluster / (len(cluster)-1) )
        
        return np.mean(np.array(variance))

    def updateCentroids(self):
        newCentroids = []
        #print(len(self.clusters))
        for cluster in self.clusters:
            hold = []
            for i in cluster:
                hold.append(self.X[i])
            
            hold = np.array(hold)
            hold = np.mean(hold , axis=0)
            
            newCentroids.append(hold)
        # I Will Explain why i did this in the video , it was really Tricky :)
        try:
            return np.array(newCentroids)
        except:
            return self.centroids

        

    def getClusters(self):
        clusters = [[] for _ in range(self.k)]
        for index , x in enumerate(self.X):
            colsestCentroidIndex = self.getMinDistance(x)
            clusters[colsestCentroidIndex].append(index)
        return clusters

    def getMinDistance(self , x):
        index = -1 
        minNumber = 1e9
        for i in range(len(self.centroids)):
            centroid = self.centroids[i]
            dist = np.sum((x-centroid)**2)**0.5
            if(dist < minNumber):
                index = i
                minNumber = dist
        return index
    
    def gerRightCluster(self,X):
        self.X = X 
        clusters = self.getClusters()
        return clusters

    
X_train ,X_test , y_train , y_test = getData()

model = Kmeans()

pred , centeriods = model.predict(X_train)

testPred  = model.gerRightCluster(X_test)

centroidsTrain = {}
centroidsTest = {}
def getLabels(pred , df, dic):
    
    for idx , data in enumerate(pred):
        for i in data:
            dic[i] = idx
    clusters = []
    for i in range(df.shape[0]):
        clusters.append(dic[i])

    return clusters

trainClusters = getLabels(pred,X_train ,centroidsTrain )
testClusters = getLabels(testPred,X_test ,centroidsTest )

plt.scatter(X_train[: , 0] , X_train[: , 1 ] , c = trainClusters)
plt.scatter(X_test[: , 0] , X_test[: , 1] , c = testClusters)
plt.scatter(centeriods[: , 0] , centeriods[:,1] , c='black' , marker = 'X')
plt.show()

