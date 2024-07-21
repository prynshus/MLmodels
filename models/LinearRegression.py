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
    def __init__(self,n_iterations=1000,lr=0.1,regularisation=0.01):
        self.n_iterations=n_iterations
        self.lr=lr
        self.regularisation=regularisation 
        self.coef_=None
        self.intercept_=None

    def fit(self,X,y):
        """
        fits the given training data in the Logistic regression model.

        Parameters:
        X (numpy.ndarray): input training set of shape (n_samples, n_features)
        y (numpy.ndarray): input training labels of shape (n_samples,)

        """

        n_instances, n_features=X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

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

    def predict_proba(self,X):
        """
        predicts probabilities of the given input data in the trained model.

        Parameters:
        X (numpy.ndarray): input data of shape (n_samples,n_features)

        return:
        numpy.ndarray: probabilities of classes
        """

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

X_train = np.array([[0.5, 1.5], [1.5, 2.5], [3.5, 0.5], [5.5, 4.5]])
y_train = np.array([0, 0, 1, 1])

model = LogisticRegression(n_iterations=100)
model.fit(X_train,y_train)

X_test = np.array([[2.0, 2.0], [4.0, 3.0]])
y_test = np.array([0, 1]) 

y_pred = model.predict(X_test)
print(y_pred)
print(model.accuracy(y_pred,y_test))