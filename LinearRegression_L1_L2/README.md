## Linear Regression

1. Uses gradient descent for loss function optimisation with convergence of 0.001.
2. eta is the learning rate which is multiplied with the gradient and n_iterations is number of iterations 
3. The input feature vector X and target y are to be given as column vectors to the model

### Cost Function

Cost function is written only for 1 feature variable although the algorithm works for any number of features
<img src="https://latex.codecogs.com/gif.latex?%5Clarge%20cost%5C%20function%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%5C%20%5B%5C%20y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%5C%20%5D%5E%7B2%7D"/>

#### Weights Updation

<img src="https://latex.codecogs.com/gif.latex?%5Clarge%20%5C%5Cfor%5C%20w_%7B1%7D%20%3A%20%5C%20w_%7B1%7D%20%3D%20w_%7B1%7D%5C%20-%20%5C%20%5Ceta%20%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%20%29%28-w_%7B1%7D%29%20%5C%5Cfor%5C%20w_%7B0%7D%20%3A%20%5C%20w_%7B0%7D%20%3D%20w_%7B0%7D%5C%20-%20%5C%20%5Ceta%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%20%29%28-1%29"/>

### Cost Function with Ridge Regularization
Here lambda is the regularization parameter

<img src="https://latex.codecogs.com/gif.latex?%5C%5Ccost%5C%20function%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%5C%20%5B%5C%20y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%5C%20%5D%5E%7B2%7D%20&plus;%20%5Cfrac%7B%5Clambda%20%7D%7B2m%7D%28w_%7B1%7D%5E%7B2%7D%29"/>

#### Weights Updation

Regularization is applied on higher order features 1 and above; bias isnt included as it has least impact on model
<img src ="https://latex.codecogs.com/gif.latex?%5Clarge%20%5C%5Cfor%5C%20w_%7B1%7D%20%3A%20%5C%20w_%7B1%7D%20%3D%20w_%7B1%7D%281%20-%20%5Cfrac%7B%5Clambda%5Ceta%20%7D%7Bm%7D%29%5C%20-%20%5C%20%5Ceta%20%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%20%29%28-w_%7B1%7D%29%20%5C%5Cfor%5C%20w_%7B0%7D%20%3A%20%5C%20w_%7B0%7D%20%3D%20w_%7B0%7D%5C%20-%20%5C%20%5Ceta%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%20%29%28-1%29"/>

### Cost Function with Lasso Regularization
Here lambda is the regularization parameter

<img src="https://latex.codecogs.com/gif.latex?%5C%5Ccost%5C%20function%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%5C%20%5B%5C%20y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%5C%20%5D%5E%7B2%7D%20&plus;%20%5Cfrac%7B%5Clambda%20%7D%7B2m%7D%7Cw_%7B1%7D%7C"/>

#### Weights Updation

Regularization is applied on higher order features 1 and above; bias isnt included as it has least impact on model
<img src ="https://latex.codecogs.com/gif.latex?%5C%5Cfor%5C%20w_%7B1%7D%20%3A%20%5C%20w_%7B1%7D%20%3D%20w_%7B1%7D%20%5Cpm%20%5Cfrac%7B%5Clambda%5Ceta%20%7D%7B2m%7D%5C%20-%20%5C%20%5Ceta%20%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%20%29%28-w_%7B1%7D%29%20%5C%5Cfor%5C%20w_%7B0%7D%20%3A%20%5C%20w_%7B0%7D%20%3D%20w_%7B0%7D%5C%20-%20%5C%20%5Ceta%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%20-%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%20%29%28-1%29"/>

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.
