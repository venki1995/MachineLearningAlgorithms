# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
 
from GradientBoost_Classification.decision_tree_modified import DecisionTreeRegressor
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('ggplot')

class GradientBoost():
    def __init__(self,n_trees,eta,max_depth):
        self.n_trees = n_trees
        self.eta = eta
        self.max_depth = max_depth
        
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            self.trees.append(tree)
        
    def fit(self,X,y):
        un,counts = np.unique(y,return_counts=True)
        log_odds = np.log(counts[1]/counts[0])
        self.initlog_odds = log_odds
        prob = np.exp(log_odds)/(1+np.exp(log_odds))
        prevprob = np.full(np.shape(y),prob)
        for i in range(self.n_trees):    
            grad = y - prevprob
            self.trees[i].fit(X,grad,prevprob)
            boost_val = self.trees[i].predict(X)
            log_odds = log_odds + self.eta*boost_val.reshape(-1,1)
            prevprob = np.exp(log_odds)/(1+np.exp(log_odds))
    
    def predict(self,X):
        wl = np.zeros(X.shape[0])
        initlog_odds = np.full((X.shape[0],1),self.initlog_odds)
        for i in range(self.n_trees):
            boost_val = self.trees[i].predict(X)
            wl = wl + self.eta*boost_val
        finlog_odds = initlog_odds + wl.reshape(-1,1)
        return np.where(np.exp(finlog_odds)/(1+np.exp(finlog_odds))>=0.5,1,0)
    

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train,y_train = X[51:150],y[51:150].reshape(-1,1)
fn = np.vectorize(lambda x: 0 if x==1 else 1)
y_train = fn(y_train)

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 2].min() - .5, X[:, 2].max() + .5
    y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_array = np.c_[xx.ravel(), yy.ravel()]
    appended_arr = np.concatenate((np.zeros((grid_array.shape[0],1)),
                                  np.zeros((grid_array.shape[0],1)),grid_array),axis=1)
    Z = pred_func.predict(appended_arr)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    fig,ax1 = plt.subplots(dpi=500)
    ax1.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
    ax1.scatter(X[:, 2], X[:, 3], c=y.ravel(),cmap=plt.cm.PiYG)
    ax1.set_xlabel("Petal length")
    ax1.set_ylabel("Petal Width")
    ax1.set(title='Gradient Boost Max_depth-{} Trees-{} LR-{}'.format(pred_func.max_depth,
                 pred_func.n_trees,pred_func.eta))

gbr = GradientBoost(n_trees=20,eta=0.4,max_depth=2)
gbr.fit(X_train,y_train)
plot_decision_boundary(gbr, X_train, y_train)