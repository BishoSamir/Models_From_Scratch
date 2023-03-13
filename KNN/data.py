from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X , y = data.data , data.target
#print(X[:3])
#print('------')
#print(y[:3])

def getData():
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.2)
    return X_train , X_test , y_train , y_test