import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn import datasets 
from Naive_Bayes import NaiveBayes

def accuracy(yTrue , yPred):
    return np.sum( (yTrue == yPred) ) / len(yTrue) * 100


X , y = datasets.make_classification(n_samples = 1000 , n_features= 8 , n_classes=2 )

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 )

model = NaiveBayes()

model.fit(X_train , y_train)

pred = model.predict(X_test)

print(f"Accuracy = {accuracy(pred , y_test)}")