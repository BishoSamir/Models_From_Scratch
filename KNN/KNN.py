import numpy as np
from data import getData
# I can Make it more efficient With Vectors form , but it will be hard to read
#  for people which i made this repo for them
# also i know i can make it more optimized but as i told you i do it for newbie people
class KNN:

    def __init__(self, k):
        self.k = k 

    def fit(self, X , y):
        self.X = X 
        self.y = y
        self.labels = self.uniqueLabels(self.y)
    
    def uniqueLabels(self,y):
        dic = {}
        for i in y:
            if(i not in dic.keys()):
                dic[i] = 0
        return dic

    def predict(self, X):
        predictions = [self.predictHelper(x) for x in X]
        self.clearDic()
        
        return predictions
    def predictHelper(self, x):
        distances = []
        for x_train , label in zip(self.X , self.y):
            distances.append((self.euclideanDistance(x_train, x) , label)) 
        distances = sorted(distances)
        
        max_label = 0
        for i in distances[:self.k]:
            self.labels[i[1]] += 1
            if(self.labels[i[1]] > max_label):
                max_label = i[1]

        return max_label
    
    def euclideanDistance(self,x1,x2):
        return np.sum( (x1-x2)**2 )**0.5
    
    def clearDic(self):
        for i in self.labels.keys():
            self.labels[i] = 0

    
X_train , X_test , y_train , y_test = getData()
model = KNN(3)
model.fit(X_train , y_train)
pred = model.predict(X_test)

print(f"acc = {np.sum(pred == y_test) / len(y_test)}")
