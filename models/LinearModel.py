import numpy as np

class LinearRegression:
    def __init__(self):
        self.coef_=None
        self.intercept_=None

    def fit(self,X,y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.pinv(X_b).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self,X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept_, self.coef_])


class LogisticRegression:
    def __init__(self,n_iterations=1000,lr=0.1,regularisation=0.01, early_stopping = False, tol = 1e-4, verbose=False, n_iter_no_change=10):
        self.n_iterations=n_iterations
        self.lr=lr
        self.regularisation=regularisation 
        self.early_stopping = early_stopping
        self.verbose=verbose
        self.tol=tol
        self.n_iter_no_change=n_iter_no_change
        self.coef_=None
        self.intercept_=None
        self.losses_ = []

    def fit(self,X,y):
        """
        fits the given training data in the Logistic regression model.

        Parameters:
        X (numpy.ndarray): input training set of shape (n_samples, n_features)
        y (numpy.ndarray): input training labels of shape (n_samples,)

        """

        X = self._normalize(X)

        n_instances, n_features=X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        best_loss = float('inf')
        no_improvement_count = 0

        for i in range(self.n_iterations):
            #linear model
            linear_model=X @ self.coef_ + self.intercept_
            #sigmoid function
            predictions = 1/(1+np.exp(-linear_model))

            #gradient descent
            error = predictions - y
            dw = (X.T @ error)/n_instances + (self.regularisation * self.coef_)/n_instances
            db = np.sum(error)/n_instances

            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db

            #calculate loss
            loss = self._computeLoss(predictions,y)
            self.losses_.append(loss)

            if self.verbose:
                print("{}/{} loss: {}".format(i+1,self.n_iterations,loss))

            if self.early_stopping:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= self.n_iter_no_change:
                        if self.verbose:
                            print("Early stopping at {} iteration.".format(i+1))
                        break

    def predict_proba(self,X):
        """
        predicts probabilities of the given input data in the trained model.

        Parameters:
        X (numpy.ndarray): input data of shape (n_samples,n_features)

        return:
        numpy.ndarray: probabilities of classes
        """
        
        X=self._normalize(X)
        linear_model = X @ self.coef_ + self.intercept_
        return 1/(1 + np.exp(-linear_model))

    def predict(self,X,threshold=0.5):
        """
        predicts class labels for the input data.

        Parameters:
        X (numpy.ndarray): input data of shape (n_samples,n_instances)
        threshold (float): classifies the probabilities by a given threshold value.

        return:
        numpy.ndarray: class labels
        """

        probability = self.predict_proba(X)
        return (probability >= threshold ).astype(int)

    def accuracy(self,y_true,y_pred):
        """
        the accuracy of the model

        Parameters:
        y_true (numpy.ndarray): actual labels of the data.
        y_pred (numpy.ndarray): predicted labels by the model.

        return:
        float: accuracy of the model.
        """

        return np.mean(y_true==y_pred)

    def _normalize(self,X):
        """
        Normalize input data.

        Parameters:
        X (numpy.ndarray): input shape of (n_samples,n_features)

        return:
        numpy.ndarray: Normalized data
        """
        return (X - X.mean(axis=0)/X.std(axis=0))

    def _computeLoss(self, y_pred, y_test):
        """
        Computes the loss.

        Parameters:
        y_pred (numpy.ndarray): predicted value of shape (n_samples,)
        y_test (numpy.ndarray): actual value of shape (n_samples,)

        return:
        float: the loss of the training
        """
        n_samples = len(y_test)
        loss = -np.mean(y_test * np.log(y_pred) + (1 - y_test)* np.log(1 - y_pred))
        return loss