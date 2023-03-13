import numpy as np 
from data import getData
class LogisticRegression:

    def __init__(self,lr = 0.01 , nIter = 1000):
        self.lr = lr
        self.nIter = nIter
        self.W = None
        self.b = None
    
    def fit(self, X, y):
        observations , features = X.shape
        self.W = np.random.randn(features)
        self.b = 0

        for _ in range(self.nIter):
            yLogit = np.dot(X , self.W.T) + self.b
            yProb = self.sigmoid(yLogit)

            dw = (1/observations) * np.dot(X.T , (yProb - y))
            db = (1/observations) * np.sum((yProb - y))
            
            self.W -= self.lr * dw
            self.b -= self.lr * db

    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X , isProb = False):
        yLogit = np.dot(X, self.W.T) + self.b
        yProb = self.sigmoid(yLogit)
        if(isProb):
            return yProb
        
        return yProb >= 0.5
    
    def getProbability(self , X ):
        yProb = self.predict(X , True)
        return yProb

def accuracy(yPred , yTrue):
    return np.mean(yPred == yTrue)

X_train , X_test , y_train , y_test = getData()

model = LogisticRegression(lr = 0.001)
model.fit(X_train , y_train)
yPred = model.predict(X_test)
yProb = model.getProbability(X_test)

print(yProb[:5] )
print(y_test[:5])
print('--------------------------------')
print(f'acc = {accuracy(yPred , y_test)}')
