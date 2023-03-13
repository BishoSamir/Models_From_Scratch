import numpy as np
from data import getData
class LinearRegression:
    
    def __init__(self,lr = 0.01 , nIter = 1000):
        self.lr = lr 
        self.nIter = nIter
        self.W = None
        self.b = None
    
    def fit(self , X ,y ):
        observations , features = X.shape
        self.W = np.random.randn(features)
        self.b = 0

        for _ in range(self.nIter):
            yPred = np.dot(X , self.W.T) + self.b
            
            dw = (1/observations) * np.dot(X.T , (yPred - y))
            db = (1/observations) * np.sum((yPred - y))

            self.W -= self.lr * dw 
            self.b -= self.lr * db

    def predict(self , X):
        return np.dot(X , self.W.T) + self.b



X_train , X_test,   y_train , y_test = getData()

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

def mse(yPred , y_test):
    return np.mean((yPred - y_test)**2)

print(f'mse = {mse(pred , y_test)}')