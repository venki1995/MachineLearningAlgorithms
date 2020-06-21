# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:25:43 2020

@author: Venkatesh
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
plt.style.use('seaborn-bright')

class LogisticRegression:
    def __init__(self,eta=0.05,n_iterations=1000,reg_parameter = 0):
        self.eta = eta
        self.n_iterations = n_iterations
        self.reg_parameter = reg_parameter
    
    def wTx(self,X):
        return np.dot(X,self.w[1:]) + self.w[0]
    
    def activation(self,z):
        return 1/(1+np.exp(-z))
    
    def fit(self,X,y):
        self.cost = []
        x = X.shape[0]
        self.w = np.random.random((X.shape[1]+1,1))
        for i in range(self.n_iterations):
            wtx = self.wTx(X)
            residuals = y - self.activation(wtx)
            self.w[1:] = self.w[1:]*(1 - (self.reg_parameter*self.eta)/x) + self.eta*np.dot(X.T,residuals)/x 
            self.w[0] = self.w[0] + self.eta*(np.sum(residuals)/x)
            self.cost.append(-y*np.log(self.activation(wtx)) - 
                             (1-y)*(np.log(1-self.activation(wtx)))+ (self.reg_parameter/(2*x))*np.sum(self.w[1:]**2))
        return self
    
    def predict(self,X):
        wtx = self.wTx(X)
        return np.where(self.activation(wtx)>=0.5,1,0)
    
    def plot_cost(self):
        fig,ax1 = plt.subplots(dpi=250)
        ax1.plot(np.arange(self.n_iterations),self.cost,label='cost',marker='.')
        ax1.set(title='Cost vs Iterations',ylabel='Cost',xlabel='n_iterations')
        ax1.legend()

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
sc = StandardScaler()
X_train = sc.fit_transform(X[:100])
y_train = y[:100].reshape(-1,1)
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    fig,ax1 = plt.subplots(dpi=500)
    ax1.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(),cmap=plt.cm.coolwarm)
    ax1.set_xlabel("Petal length_standardised")
    ax1.set_ylabel("Petal Width_standardised")
    ax1.set_title("Binary classification of Iris Dataset - Class 0,1")

lr = LogisticRegression(reg_parameter = 0)
lr.fit(X_train,y_train)
plot_decision_boundary(lr.predict, X_train, y_train)
