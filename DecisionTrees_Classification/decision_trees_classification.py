# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:25:14 2020

@author: ven10
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('seaborn-bright')

class DecisionTreeClassifier():
    def __init__(self,max_depth=5,depth=1):
        self.max_depth = max_depth
        self.depth = depth
    
    def gini_impurity(self,array):
        _,counts = np.unique(array,return_counts=True)
        counts_prob = counts/len(array)
        return 1 - np.sum(np.square(counts_prob))
        
    def info_gain(self,l_cnt,left_gini,r_cnt,right_gini,gini_node):
        return gini_node - (l_cnt/(l_cnt+r_cnt))*left_gini - (r_cnt/(l_cnt+r_cnt))*right_gini
    
    def find_best_split(self,col,y):
        info_gain = 0
        split = None
        gini_node = self.gini_impurity(y)
        sorted_col,labels = list(zip(*sorted(zip(col,y))))
        sorted_col = np.array(sorted_col)
        labels = np.array(labels)
        thres = (sorted_col[1:] + sorted_col[:-1])/2
        
        for vals in np.unique(thres):
            left = labels[col < vals]
            right = labels[col >= vals]
            left_gini = self.gini_impurity(left)
            right_gini = self.gini_impurity(right)
            score = self.info_gain(len(left),left_gini,len(right),right_gini,gini_node)
            if score > info_gain and info_gain >= 0:
                info_gain = score
                split = vals
        return split,info_gain
     
    def find_best_split_all(self,X,y):
        best_split = {}
        for i,c in enumerate(X.T):
            split_c,info_gain = self.find_best_split(c, y)
            if not best_split or best_split['info_gain'] < info_gain:
                best_split = {"split":split_c,'info_gain':info_gain,"col_index":i}
        return best_split["split"],best_split["col_index"]
    
    def fit(self,X,y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self.grow_tree(X,y)
        
    def grow_tree(self,X,y,depth=0):
        uns,counts = np.unique(y,return_counts=True)
        label_array = np.asarray(list(zip(uns,counts)))
        pred_class = label_array[np.argmax(label_array[:,1]),0]
        node = TreeNode(self.gini_impurity(y),len(y),pred_class)
        if depth < self.max_depth:
            split_val,split_col = self.find_best_split_all(X, y)
            if split_val is not None:
                left_X,left_y = X[X[:,split_col] <= split_val],y[X[:,split_col] <= split_val]
                right_X,right_y = X[X[:,split_col] > split_val],y[X[:,split_col] > split_val]
                node.feat_indx = split_col
                node.split_val = split_val
                node.left = self.grow_tree(left_X,left_y,depth+1)
                node.right = self.grow_tree(right_X, right_y,depth+1)
        return node
    
    def predict(self,X):
        return np.array([self._predict(inputs) for inputs in X])
        
    def _predict(self,inputs):
        node = self.tree
        while node.left:
            if inputs[node.feat_indx] <= node.split_val:
                node = node.left
            else:
                node = node.right
        return node.pred_class
    
class TreeNode:
    def __init__(self,gini,samples,pred_class):
        self.gini = gini
        self.samples = samples
        self.pred_class = pred_class
        self.feat_indx = None
        self.split_val = None
        self.left = None
        self.right = None


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train,y_train = X[:150],y[:150]
X_test,y_test = X[80:100],y[80:100]

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 2].min() - .5, X[:, 2].max() + .5
    y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_array = np.c_[xx.ravel(), yy.ravel()]
    appended_arr = np.concatenate((np.zeros((grid_array.shape[0],1)),
                                  np.zeros((grid_array.shape[0],1)),grid_array),axis=1)
    Z = pred_func(appended_arr)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    fig,ax1 = plt.subplots(dpi=500)
    ax1.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
    ax1.scatter(X[:, 2], X[:, 3], c=y.ravel(),cmap=plt.cm.PiYG)
    ax1.set_xlabel("Petal length")
    ax1.set_ylabel("Petal Width")

dt_classifier = DecisionTreeClassifier(max_depth=3)
dt_classifier.fit(X_train,y_train)
ypred = dt_classifier.predict(X_test)
plot_decision_boundary(dt_classifier.predict, X_train, y_train)
        