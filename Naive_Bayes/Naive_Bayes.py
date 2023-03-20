import numpy as np 

class NaiveBayes:

    def fit(self , X , y):
        observations , features = X.shape
        self.classes = np.unique(y)

        self.mean = np.zeros((len(self.classes) , features), dtype=np.float64)
        self.var = np.zeros((len(self.classes) , features), dtype=np.float64)
        self.priors = np.zeros(len(self.classes) , dtype=np.float64)
        
        for i in self.classes:
            X_PerClass = X[i == y]
            self.mean[i , :] = np.mean(X_PerClass , axis = 0)
            self.var[i , :] = np.var(X_PerClass , axis =0)
            self.priors[i] = len(X_PerClass) / float(observations)

    
    def predict(self , X ):
        yPred = [self.predictHelper(x) for x in X]
        return yPred


    def predictHelper(self , x):
        ans = -1e9
        answerProb = -1e9

        for i in self.classes:
            mean = self.mean[i]
            var = self.var[i]
            pdf =  np.exp( -(x-mean)**2 / (2*var) ) / (2 * np.pi * var)**0.5
            #print(pdf)
            pdf = np.sum(np.log(pdf)) + np.log( self.priors[i] )
            if(pdf > answerProb):
                ans = i
                answerProb = pdf
        
        return ans