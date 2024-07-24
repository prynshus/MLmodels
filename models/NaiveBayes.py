import numpy as np

class naiveBayes:
    def __init__(self,standardize=False):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.prior = {}
        self.standardize=standardize

    def fit(self,X,y):
        X = self._impute_missing_values(X)
        self.classes = np.unique(y)
        if self.standardize:
            X=self._standardize(X)
        for c in self.classes:
            x_c = X[c == y]
            self.mean[c] = x_c.mean(axis=0)
            self.var[c] = x_c.var(axis=0)
            self.prior[c] = x_c.shape[0] / X.shape[0]

    def _calculateLikelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x - mean)**2 /(2*var))
        denominator = np.sqrt(2* np.pi * var)
        return numerator / denominator 
    
    def _calculatePrior(self,class_idx):
        return np.log(self.prior[class_idx])

    def _classify(self, x):
        posteriors=[]
        for c in self.classes:
            prior = self._calculatePrior(c)
            likelihood = np.sum(np.log(self._calculateLikelihood(c,x)))
            posterior = prior + likelihood 
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self,X):
        X = self._impute_missing_values(X)
        if self.standardize:
            X=self._standardize(X)
        return [self._classify(x) for x in X]
    
    def _standardize(self,X):
        return ((X - X.mean(axis=0))/X.std(axis=0))

    def _impute_missing_values(self,X):
        if hasattr(X, 'toarray'):  # Check if X is a sparse matrix
            X = X.toarray()
        col_mean = np.nanmean(X, axis=0)
        i = np.where(np.isnan(X))
        X[i] = np.take(col_mean, i[1])
        return X