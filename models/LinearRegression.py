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


class decisionTree:
    def __init__(self,max_depth = 5, min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.tree = None
        self.class_mapping = None # If the class labels are not sequential or do not start from 0, this would cause an issue.

    def fit(self,X,y):
        """
        fits the data in the decision tree

        Parameters:
        X (numpy.ndarray): input shape (n_samples,n_features)
        """
        self.class_mapping = {cls: idx for idx, cls in enumerate(np.unique(y))}
        self.tree = self._growtree(X,y)

    def predict(self,X):
        """
        predicts the given input.

        Parameters:
        X (numpy.ndarray): input shape (n_samples,n_features)

        return:
        numpy.ndarray: returns the label.
        """
        return np.array([self._predict(x, self.tree) for x in X])

    def _split(self,X,y,idx,t):
        left= np.where(X[:,idx]<=t)
        right=np.where(X[:,idx]>t)
        return (X[left], y[left]), (X[right], y[right])

    def _bestSplit(self, X,y):
        m,n = X.shape
        if m<=1:
            return None, None
        
        num_parent = [np.sum(y == c) for c in self.class_mapping.keys()]
        best_gini = 1.0 - sum((i/m) **2 for i in num_parent)
        best_idx, best_t = None, None

        for idx in range(n):
            threshold, classes = zip(*sorted(zip(X[:,idx],y)))
            num_left = [0] * len(num_parent)
            num_right=num_parent.copy()

            for i in range(1,m):
                c_i = classes[i-1]
                c = self.class_mapping[c_i]
                num_left[c] += 1
                num_right[c]-= 1
                gini_left = 1.0 - sum((num_left[x]/i) ** 2 for x in np.unique(y))
                gini_right= 1.0 - sum((num_right[x]/(m-i)) ** 2 for x in np.unique(y))
                gini = (i* gini_left + (m-i)* gini_right)/m
                if threshold[i] == threshold[i-1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_t = (threshold[i] + threshold[i -1])/2
                    best_idx= idx
        return best_idx, best_t

    def _growtree(self,X,y,depth=0):
        num_samples_per_class = [np.sum(y == c) for c in self.class_mapping.keys()]
        predicted = np.argmax(num_samples_per_class)
        node = {"Predicted" : list(self.class_mapping.keys())[predicted]}

        if depth<self.max_depth:
            idx, t = self._bestSplit(X,y)
            if idx is not None:
                (X_left, y_left), (X_right, y_right) = self._split(X,y,idx,t)
                if len(y_left) >= self.min_samples_split and len(y_right) >= self.min_samples_split:
                    node["feature_index"] = idx
                    node["threshold"] = t
                    node["left"] = self._growtree(X_left,y_left,depth+1)
                    node["right"]= self._growtree(X_right,y_right,depth+1)
          
        return node

    def _predict(self,x,tree):
        if "threshold" in tree:
            if x[tree["feature_index"]] <= tree["threshold"]:
                return self._predict(x, tree["left"])
            else:
                return self._predict(x, tree["right"])
        else:
            return tree["Predicted"]

    def accuracy(self,y_pred,y):
        """
        calculates accuracy.

        Parameters:
        y_pred (numpy.ndarray): predicted value by the model (n_samples,)
        y (nump.ndarray): actual data of shape (n_samples,)

        return:
        float: accuracy.
        """
        return sum(y_pred == y)/y.shape[0]