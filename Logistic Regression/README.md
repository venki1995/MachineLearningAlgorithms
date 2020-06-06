
## Logistic Regression

1. Uses gradient descent for loss function optimisation
2. eta is the learning rate which is multiplied with the gradient and n_iterations is number of iterations 
3. The input feature vector X and target y are to be given as column vectors to the model

### Cost Function

Cost function is written only for 1 feature variable although the algorithm works for any number of features
<img src="https://latex.codecogs.com/gif.latex?cost%5C%20function%20%3D%20%5Cfrac%7B-1%7D%7B2m%7D%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%5C%20%5B%5C%20%28y%5E%7B%28i%29%7D%20%5C%20log%28%5Csigma%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%29%5C%20&plus;%20%281%20-%20y%5E%7B%28i%29%7D%29%20%5C%20%281-log%28%5Csigma%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%29%20%5D%20&plus;%20%5Cfrac%7B%5Clambda%20%7D%7B2m%7D%28w_%7B1%7D%5E%7B2%7D%29"/>

#### Weights Updation

<img src="https://latex.codecogs.com/gif.latex?%5C%5Cfor%5C%20w_%7B1%7D%20%3A%20%5C%20w_%7B1%7D%20%3D%20w_%7B1%7D%281%20-%20%5Cfrac%7B%5Clambda%5Ceta%20%7D%7Bm%7D%29%5C%20&plus;%20%5C%20%5Ceta%20%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%20-%20%5Csigma%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%20%29%28w_%7B1%7D%29%20%5C%5Cfor%5C%20w_%7B0%7D%20%3A%20%5C%20w_%7B0%7D%20%3D%20w_%7B0%7D%5C%20&plus;%20%5C%20%5Ceta%5Cfrac%7B1%7D%7Bm%7D%5C%20%5Csum%20_%7Bi%3D1%7D%5E%7Bm%7D%28y%5E%7B%28i%29%7D%20-%20%5Csigma%20%28w_%7B0%7D%20&plus;%20w_%7B1%7Dx%5E%7B%28i%29%7D%29%20%29"/>


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.
