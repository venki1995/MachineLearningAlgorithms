# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:55:30 2020

@author: ven10
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as sklearn_SVC
plt.style.use('seaborn-bright')

class SVM:
    def __init__(self,eta=0.005,n_iterations=2000,reg_parameter = 0.1):
        self.eta = eta
        self.n_iterations = n_iterations
        self.reg_parameter = reg_parameter
    
    def wTx(self,X):
        return np.dot(X,self.w)
  
    def fit(self,X,y):
        self.cost = []
        m,n = X.shape
        seed = np.random.RandomState(42)
        self.w = seed.random((n,1))
        for epoch in range(1,self.n_iterations):
            hinge = y*self.wTx(X)
            yixi = y*X
            grad = self.w - (self.reg_parameter*np.sum(yixi[np.ravel(hinge<1),:],axis=0).reshape(-1,1))
            self.w = self.w - self.eta*grad
            self.cost.append(np.sum(self.w**2) + np.sum(hinge[hinge<1])) 
    
    def predict(self,X):
        return np.sign(self.wTx(X))
    
    def plot_cost(self):
        fig,ax1 = plt.subplots(dpi=250)
        ax1.plot(np.arange(1,self.n_iterations),self.cost,label='cost',marker='.')
        ax1.set(title='Cost vs Iterations',ylabel='Cost',xlabel='n_iterations')
        ax1.legend()

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
sc = StandardScaler()
X_train = sc.fit_transform(X[:100])
X_train = np.append(X_train,[[-1.0,0.0],[-1.5,0.0]],axis=0)
X_train = np.hstack([X_train,np.ones(X_train.shape[0]).reshape(-1,1)])

y_train = y[:100].reshape(-1,1)
y_train = np.append(y_train,[[1],[1]],axis=0)
y_train[y_train==0]=-1

def plot_decision_boundary(pred_func, X, y,reg):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func.predict(np.c_[xx.ravel(), yy.ravel(),np.ones(xx.ravel().shape)])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    fig,ax1 = plt.subplots(dpi=500)
    ax1.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(),cmap=plt.cm.coolwarm)
    ax1.set_xlabel("Petal length_standardised")
    ax1.set_ylabel("Petal Width_standardised")
    ax1.set_title("Binary classification of Iris Dataset C-{}".format(reg))

svm = SVM()
svm.fit(X_train,y_train)
plot_decision_boundary(svm, X_train, y_train,svm.reg_parameter)
#svm.plot_cost()

#scikit_svm = sklearn_SVC(kernel="linear",C=0.1)
#scikit_svm.fit(X_train,y_train)
#plot_decision_boundary(scikit_svm, X_train, y_train,0.1)