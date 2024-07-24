import numpy as np
import graphviz

class decisionTree:
    def __init__(self,max_depth = 5, min_samples_split=2,min_impurity_decrease=0.0):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None
        self.feature_importances_ = None
        self.class_mapping = None # If the class labels are not sequential or do not start from 0, this would cause an issue.

    def fit(self,X,y):
        """
        fits the data in the decision tree

        Parameters:
        X (numpy.ndarray): input shape (n_samples,n_features)
        """
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
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

    def _gini(self,y):
        m=len(y)
        if m==0:
            return 0
        return 1.0 - sum((np.sum(y==c)/m)**2 for c in np.unique(y))

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
                if gini < best_gini and (best_gini - gini) >= self.min_impurity_decrease:
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
                    impurity_decrease = self._gini(y) - (len(y_left) / len(y) * self._gini(y_left) + len(y_right) / len(y) * self._gini(y_right))
                    self.feature_importances_[idx] += impurity_decrease
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

    def feature_importances(self):
        total_importance = np.sum(self.feature_importances_)
        return self.feature_importances_ / total_importance if total_importance != 0 else np.zeros_like(self.feature_importances_)

    def _export_graphviz(self,node,depth=0):
        if threshold in node:
            left_label = self._export_graphviz(node["left"],depth+1)
            right_label = self._export_graphviz(node["right"],depth+1)
            return 'feature_{feature_index} <= {threshold} ?\nleft -> {left_label}\nright -> {right_label}\n'.format(feature_index=node["feature_index"],
                                                                                                                     threshold=node["threshold"],
                                                                                                                     left_label=left_label,
                                                                                                                     right_label=right_label
                                                                                                                    )

        else:
            return "class: {}".format(node["Predicted"])   

    def export_graphviz(self):
        return "digraph tree: \n{}".format(self._export_graphiz(self.tree))        

    def visualize(self, feature_names=None, class_names=None):
        dot_data = self._to_graphviz(self.tree, feature_names, class_names)
        graph = graphviz.Source(dot_data)
        graph.render("decision_tree", format='png', cleanup=True)                     

    def _to_graphviz(self, tree, feature_names=None, class_names=None, depth=0, node_id=0):
        nodes = []
        edges = []
        self._add_node(tree, nodes, edges, feature_names, class_names, node_id)
        dot_data = 'digraph Tree {\n'
        dot_data += 'node [shape=box, style="filled", color="black", fontname="helvetica"] ;\n'
        dot_data += 'edge [fontname="helvetica"] ;\n'
        dot_data += '\n'.join(nodes)
        dot_data += '\n'.join(edges)
        dot_data += '\n}'
        return dot_data  

    def _add_node(self, tree, nodes, edges, feature_names, class_names, node_id, parent_id=None, label=None):
        if "threshold" in tree:
            feature = feature_names[tree["feature_index"]] if feature_names else f'X[{tree["feature_index"]}]'
            nodes.append(f'{node_id} [label="{feature} <= {tree["threshold"]}\nGini={self._gini_str(tree)}", fillcolor="#e58139"] ;')
            if parent_id is not None:
                edges.append(f'{parent_id} -> {node_id} [labeldistance=2.5, labelangle=45, headlabel="{label}"] ;')
            left_id = node_id + 1
            right_id = self._add_node(tree["left"], nodes, edges, feature_names, class_names, left_id, node_id, 'yes')
            return self._add_node(tree["right"], nodes, edges, feature_names, class_names, right_id, node_id, 'no')
        else:
            class_name = class_names[tree["Predicted"]] if class_names else tree["predicted_class"]
            nodes.append(f'{node_id} [label="class: {class_name}", fillcolor="#39e581"] ;')
            if parent_id is not None:
                edges.append(f'{parent_id} -> {node_id} [labeldistance=2.5, labelangle=-45, headlabel="{label}"] ;')
            return node_id + 1                                                          

    def _gini_str(self, tree):
        if "left" in tree:
            if "y" in tree["left"]:  # Check if 'y' exists in the left subtree
                return str(self._gini(tree["left"]["y"]))
            else:
                return "N/A"  # Handle the case where 'y' is missing
        elif "y" in tree:  # Check if 'y' exists in the current node
            return str(self._gini(tree["y"]))
        else:
            return "N/A"  # Handle the case where 'y' is missing


class RandomForest:
    def __init__(self,n_estimators=10,max_features="sqrt",max_depth=5,min_samples_split=2,min_impurity_decrease=0.0,random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.trees = []
        self.random_state=random_state
        if self.random_state is not None:
            np.random.seed(random_state)

    def _build_tree(self, X,y):
        n_samples, n_features = X.shape
        selected_features = self._max_features(n_features)
        indices = np.random.choice(n_samples,n_samples,replace = True)
        features = np.random.choice(n_features, selected_features, replace = False)
        X_sample = X[indices][:,features]
        y_sample = y[indices]
        tree = decisionTree(max_depth=self.max_depth,min_samples_split=self.min_samples_split,min_impurity_decrease=self.min_impurity_decrease)
        tree.fit(X_sample,y_sample)
        tree.features = features
        return tree

    def fit(self,X,y):
        self.trees = [self._build_tree(X,y) for _ in range(self.n_estimators)]

    def _max_features(self,n_features):
        if isinstance(self.max_features,int):
            return self.max_features
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        else:
            return n_features

    def _predict_tree(self,tree,X):
        return [tree._predict(inputs, tree.tree) for inputs in X]

    def predict(self,X):
        predicted = [self._predict_tree(tree,X) for tree in self.trees]
        return np.array([np.bincount(i).argmax() for i in zip(*predicted)])