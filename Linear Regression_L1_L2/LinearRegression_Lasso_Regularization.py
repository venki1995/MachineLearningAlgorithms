# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:16:57 2020

@author: Venkatesh
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-bright")

class LassoRegression:
    def __init__(self,eta=0.001,n_iterations=1000,reg_parameter=0):
        self.eta = eta
        self.n_iterations = n_iterations
        self.reg_parameter = reg_parameter
    
    def predict(self,X):
        return np.dot(X,self.w[1:]) + self.w[0]  #using the fit intercept term
     
    def fit(self,X,y):
        self.cost = []
        x = X.shape[0]
        random_state = np.random.RandomState(45)
        self.w = random_state.random((X.shape[1]+1,1))
        for i in range(self.n_iterations):
            residuals = y - self.predict(X)
            self.w[1:] = self.w[1:] - np.sign(self.w[1:])*(self.reg_parameter*self.eta)/(2*x) +self.eta*(np.dot(X.T,residuals)/x) #loss function = 1/2m(y-ypred)^2
            self.w[0] = self.w[0] + self.eta*(np.sum(residuals)/x)  #since bias term never affects
            self.cost.append(np.sum(residuals**2)/2*x + self.reg_parameter*np.sum(np.abs(self.w[1:]))/2*x)
        return self
    
    def plot_cost(self):
        fig,ax1 = plt.subplots(dpi=250)
        ax1.plot(np.arange(self.n_iterations),self.cost,label='cost',marker='.')
        ax1.set(title='Cost vs Iterations',ylabel='Cost',xlabel='n_iterations')
        ax1.legend()
    
x = np.arange(0,10,0.5).reshape(-1,1)       #input x signals
seed = np.random.RandomState(42)
y = 0.5*x + 3*seed.random(x.shape)  #input y signals

fig1,ax1 = plt.subplots(dpi=250)
fig2,ax2 = plt.subplots(dpi=250)
ax1.scatter(x,y,label='Actual')

for i in [0,10,100,1000]:
    lasso = LassoRegression(reg_parameter=i)
    lasso.fit(x,y)
    lasso_pred = lasso.predict(x)
    ax1.plot(x,lasso_pred,marker='.',label='Predicted_theta '+str(i))
    ax1.set(title='Linear Regression',ylabel='y values',xlabel='x values')
    ax1.legend()
   
    ax2.scatter(lasso.w[0],lasso.w[1],marker='o',label='Weights_theta '+str(i))
    ax2.set(title='Weights',ylabel='w1 weights',xlabel='w0 weights')
    ax2.legend()
    


