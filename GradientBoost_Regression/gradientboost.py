# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
 
from DecisionTrees_Regression.decision_tree_regression import DecisionTreeRegressor
import numpy as np 
import matplotlib.pyplot as plt
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
        ypred = np.full(np.shape(y),np.mean(y))
        self.initpred = np.mean(y)
        for i in range(self.n_trees):    
            grad = y - ypred
            self.trees[i].fit(X,grad)
            boost_val = self.trees[i].predict(X)
            ypred = ypred + self.eta*boost_val.reshape(-1,1)
    
    def predict(self,X):
        wl = np.zeros(X.shape[0])
        for i in range(self.n_trees):
            boost_val = self.trees[i].predict(X)
            wl = wl + self.eta*boost_val
        return np.full((X.shape[0],1),self.initpred) + wl.reshape(-1,1)
    
seed = np.random.RandomState(42)

X = np.linspace(0,1,50)
y = np.square(X) + 0.2*seed.random((50))

gbr = GradientBoost(n_trees=20,eta=0.1,max_depth=1)
gbr.fit(X.reshape(-1,1),y.reshape(-1,1))
ypred = gbr.predict(X.reshape(-1,1))
fig,ax1 = plt.subplots(dpi=300,figsize=(11,6))
ax1.plot(X,np.ravel(ypred),color="red",label="y_predicted")
ax1.scatter(X,y,label="yactual",color="green")
ax1.set(title='Gradient Boost Max_depth-{} Trees-{} LR-{}'.format(gbr.max_depth,
                 gbr.n_trees,gbr.eta),ylabel='y values',xlabel='x values')
ax1.legend()
