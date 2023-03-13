import numpy as np 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X ,y = make_blobs(centers=3 , n_samples= 1000 , n_features=2 , shuffle=True , random_state=42 )
#print(X[:2])
#print(y[:2])
#plt.scatter(X[: , 0] , X[: , -1] , c=y)
#plt.show()
def getData():
    X_train , X_test ,y_train , y_test = train_test_split(X,y , test_size = 0.2 )
    return X_train , X_test , y_train , y_test



#lst = []
#for x in X[:5] :
#    lst.append(x)
#print(np.mean(np.array(lst) , axis = 0))
#print(X[:5 , 0].mean())
#    

#arr1 = np.array([1,2,3])
#arr2 = np.array([1,2,3])
#print((arr1 == arr2).all())
#print(np.unique(y))
#lst = [[1,2] , [3,4] , np.nan ]
#lst = [ False if str(i)=='nan' else True for i in lst  ]
#lst = np.array(lst)
#print(lst.all())