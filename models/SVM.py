import numpy as np

class SVM:
    def __init__(self,n_iterations=100,lr=0.1,C=1.0,lambda_param=0.01,kernel='linear',gamma='scale',verbose=True):
        self.n_iterations=n_iterations
        self.C=C
        self.verbose=verbose
        self.kernel=kernel
        self.gamma=gamma
        self.lambda_param=lambda_param
        self.lr=lr
        self.weights_=None
        self.bias_=None

    def _linearKernel(self,X1,X2):
        return np.dot(X1,X2.T)

    def _gaussianKernel(self,X1,X2):
        if self.gamma == 'scale':
            self.gamma = 1/X1.shape[1]
        K = np.zeros(X1.shape[0],X2.shape[0])
        for i in X1.shape[0]:
            for j in X2.shape[0]:
                K[i,j] = np.exp(-self.gamma * np.linalg.norm(X1[i] - X2[j])**2)
        return K

    def _kernel(self, X1, X2):
        if self.kernel == 'linear':
            self._linearKernel(X1,X2)
        elif self.kernel == 'rbf':
            self._gaussianKernel(X1,X2)
        else: raise ValueError("Unsupported kernel")

    def fit(self,X,y):
        """
        fits the given data in the SVM model.

        Parameters:
        X (numpy.ndarray): input shape of (n_samples,n_features)
        y (numpy.ndarray): input shape of (n_samples,)
        """
        n_samples, n_features = X.shape
        y_ = np.where(y>=0,1-1)

        #K = self._kernel(X,X)

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0

        for _ in range(self.n_iterations):
            for i, x_i in enumerate(X):
                condition = y_[i] *((np.dot(x_i, self.weights_) - self.bias_) >= 1)
                if condition:
                    self.weights_ -= self.lr * (2 * self.weights_ * self.lambda_param)
                else:
                    self.weights_ -= self.lr * (2 * self.weights_ * self.lambda_param - np.dot(x_i, y_[i]))
                    self.bias_    -= self.lr * y_[i]

    def predict(self,X):
        """
        predicts the given data.

        Parameters:
        X (numpy.ndarray): input shape of (n_samples,n_features)

        return:
        numpy.ndarray: predicted value.
        """
        approx = np.dot(X,self.weight_) - self.bias
        return np.sign(approx)