# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:15:54 2020

@author: venkatesh
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class NeuralNets():
    def __init__(self,n_epochs,n_hidden,l2=0,seed=42,eta=0.01):
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden
        self.l2 = l2
        self.random = np.random.RandomState(seed)
        self.eta = eta
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def feedforward(self,X):
        z_h = np.dot(X,self.w_h) + self.b_h
        a_h = self.sigmoid(z_h)
        z_out = np.dot(a_h,self.w_out) + self.b_out
        a_out = self.sigmoid(z_out)
        return z_h,a_h,z_out,a_out
    
    def predict(self,X):
        z_h,a_h,z_out,a_out = self.feedforward(X)
        return np.where(a_out>0.5,1,0)
        
    def fit(self,X_train,y_train,X_valid,y_valid):
        self.eval_cost = []
        self.w_h = self.random.normal(loc=0.0,scale=0.05,size=(X_train.shape[1],self.n_hidden))
        self.b_h = np.zeros(self.n_hidden)
        self.w_out = self.random.normal(loc=0.0,scale=0.05,size=(self.n_hidden,1))
        self.b_out = np.zeros(1)
        
        for i in range(self.n_epochs):
            z_h,a_h,z_out,a_out = self.feedforward(X_train)
            delta_out = a_out - y_train
            
            #updating hidden layer weights
            delta_h = np.dot(delta_out,self.w_out.T)*a_h*(1-a_h)
            grad_w_h = np.dot(X_train.T,delta_h)
            grad_b_h = np.sum(delta_h,axis=0)
            grad_w_out = np.dot(a_h.T,delta_out)
            grad_b_out = np.sum(delta_out,axis=0)
            
            delta_w_h = grad_w_h + self.l2*self.w_h
            
            self.w_h -= self.eta*delta_w_h
            self.b_h -= self.eta*grad_b_h
                        
            #updating output layer weights
            
            delta_w_out = grad_w_out + self.l2*self.w_out
            self.w_out -= self.eta*delta_w_out
            self.b_out -= self.eta*grad_b_out
            
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = np.sum(y_train==y_train_pred).astype(float)/X_train.shape[0]
            val_acc = np.sum(y_valid==y_valid_pred).astype(float)/X_valid.shape[0]
            
            self.eval_cost.append(np.sum(self.cost(y_train,a_out)))
            print("Epochs: %.f,training_acc: %.2f,valid_acc: %.2f" %(i,train_acc,val_acc))
            
    def cost(self,y,output):
        l2_term = self.l2*(np.sum(self.w_h**2) + np.sum(self.w_out**2))
        cost = -y*np.log(output)- (1-y)*np.log(1-output)
        cost_l2 = cost + l2_term
        return cost_l2

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
    ax1.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
    ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(),cmap=plt.cm.PiYG)
    #fig.show()
    
X,y = datasets.make_moons(n_samples=300,shuffle=True,noise=0.05,random_state=120)
y = y.reshape(-1,1)
X_normalised = (X - np.mean(X,axis=0))/np.std(X,axis=0)


X_train,y_train = X_normalised[:200],y[:200]
X_valid,y_valid = X_normalised[200:300],y[200:300]

nn = NeuralNets(n_epochs=100,n_hidden=10,l2=0.1,eta=0.1)
nn.fit(X_train,y_train,X_valid,y_valid)
#plt.plot(range(nn.n_epochs),nn.eval_cost)
plot_decision_boundary(nn.predict, X, y)