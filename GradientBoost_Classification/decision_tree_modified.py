# -*- coding: utf-8 -*-

"""
Created on Thu Jun 18 12:25:14 2020

@author: ven10
"""
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('ggplot')

class DecisionTreeRegressor():
    def __init__(self,max_depth=5):
        self.max_depth = max_depth
    
    def MSE(self,array):
        return np.var(array)
        
    def info_gain(self,l_cnt,left_gini,r_cnt,right_gini,gini_node):
        return gini_node - (l_cnt/(l_cnt+r_cnt))*left_gini - (r_cnt/(l_cnt+r_cnt))*right_gini
    
    def find_best_split(self,col,y):
        info_gain = 0
        split = None
        gini_node = self.MSE(y)
        sorted_col,labels = list(zip(*sorted(zip(col,y))))
        sorted_col = np.array(sorted_col)
        labels = np.array(labels)
        thres = (sorted_col[1:] + sorted_col[:-1])/2
        
        for vals in np.unique(thres):
            left = labels[col < vals]
            right = labels[col >= vals]
            left_gini = self.MSE(left)
            right_gini = self.MSE(right)
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
    
    def fit(self,X,y,prevprob):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self.grow_tree(X,y,prevprob)
        
    def grow_tree(self,X,y,prevprob,depth=0):
        sumprob = np.sum(prevprob*(1-prevprob))
        pred_val = np.sum(y)/sumprob
        node = TreeNode(self.MSE(y),len(y),pred_val)
        if depth < self.max_depth:
            split_val,split_col = self.find_best_split_all(X, y)
            if split_val is not None:
                left_X,left_y,left_prob = X[X[:,split_col] <= split_val],y[X[:,split_col] <= split_val],prevprob[X[:,split_col] <= split_val]
                right_X,right_y,right_prob = X[X[:,split_col] > split_val],y[X[:,split_col] > split_val],prevprob[X[:,split_col] > split_val]
                node.feat_indx = split_col
                node.split_val = split_val
                node.left = self.grow_tree(left_X,left_y,left_prob,depth+1)
                node.right = self.grow_tree(right_X, right_y,right_prob,depth+1)
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
        return node.pred_val
    
class TreeNode:
    def __init__(self,gini,samples,pred_val):
        self.gini = gini
        self.samples = samples
        self.pred_val = pred_val
        self.feat_indx = None
        self.split_val = None
        self.left = None
        self.right = None
        
'''
seed = np.random.RandomState(42)

X = np.linspace(0,1,50)
y = np.square(X) + 0.2*seed.random((50))

dtr = DecisionTreeRegressor(max_depth=2)
dtr.fit(X.reshape(-1,1),y.reshape(-1,1))
ypred = dtr.predict(X.reshape(-1,1))
fig,ax1 = plt.subplots(dpi=300)
ax1.plot(X,np.ravel(ypred),color="red",label="y_predicted")
ax1.scatter(X,y,label="yactual",color="green")
ax1.set(title='Decision Tree Regression_Max Depth_2',ylabel='y values',xlabel='x values')
ax1.legend()

'''