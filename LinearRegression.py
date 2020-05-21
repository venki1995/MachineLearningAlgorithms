import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('seaborn-bright')

class LinearRegression:
    def __init__(self,eta=0.05,n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations
    
    def predict(self,X):
        return np.dot(X,self.w[1:]) + self.w[0]  #using the fit intercept term
    
    def fit(self,X,y):
        self.cost = []
        x = X.shape[0]
        self.w = np.random.random((X.shape[1]+1,1))
        for i in range(self.n_iterations):
            residuals = y - self.predict(X)
            self.w[1:] = self.w[1:] + self.eta*(np.dot(X.T,residuals)/x) #loss function = 1/2m(y-ypred)^2
            self.w[0] = self.w[0] + self.eta*(np.sum(residuals)/x)
            self.cost.append(np.sum(residuals**2)/2*x)
            if np.abs(np.sum(residuals)) < 10**-3: #convergence criteria
                self.convergence = i
                break
            else:
                self.convergence = i
        return self
    
    def plot_cost(self):
        fig,ax1 = plt.subplots(dpi=250)
        ax1.plot(np.arange(self.convergence+1),self.cost,label='cost',marker='.')
        ax1.set(title='Cost vs Iterations',ylabel='Cost',xlabel='n_iterations')
        ax1.legend()
    
x = np.arange(0,10,0.5).reshape(-1,1)       #input x signals
random_state = np.random.RandomState(42)
y = 0.5*x + 3*random_state.random(x.shape)  #input y signals

lm = LinearRegression()
lm.fit(x,y)
ypred = lm.predict(x)

fig,ax = plt.subplots(dpi=250)
ax.scatter(x,y,label='Actual')
ax.plot(x,ypred,marker='.',c='g',label='Predicted')
ax.set(title='Linear Regression',ylabel='y values',xlabel='x values')
ax.legend()

lm.plot_cost()

print("Final w values obtained are \n",lm.w)



