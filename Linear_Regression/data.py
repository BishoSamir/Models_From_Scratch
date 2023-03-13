import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X , y = make_regression(n_samples = 1000 , n_features= 3 , noise = 10 )

def getData():
    X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size = 0.2)
    return X_train , X_test , y_train , y_test
