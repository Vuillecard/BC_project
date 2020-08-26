# Bayesian computation project 
#### Pierre Vuillecard 
The goal of this project was to implement Bayesian model and sampling method seen in class to real data.

#### Implementation :
The code was implemented in python using Numpy, Scipy.stats and PyTorch to compute gradient. 

#### Dataset :
The dataset was found on Kaggle :
 - https://www.kaggle.com/quantbruce/real-estate-price-prediction . It consists of real estate dataset. there are 6 predictors to model the Price per unit in a city 
 - */Real estate.csv* : CSV file that contains all the data.

#### Code :
The code is decompose into 7 classes :
- The models definition classes :
    - */Bayesian_framework.py* : Object that describes the model.
    - */Model.py* : model definition.
    - */Model_torch.py* : model definition with PyTorch implementation.
- The sampling and approximation classes :
    - */Metropolis_hastings.py* : Implementation of the Metropolis Hastings algorithm in order to sample from a distribution.
    - */GVA* : Implementation of the Gaussian variational approximation algorithm in order to approximate a distribution.
    - */Important_sampling.py* : Implementation of the important sampling algorithm in order to sample from a distribution.
- Vizualisation classes :
    - */Vizualisation_tool.py* : Statistic and vizualisation tool in order to access the goodness of the model.

#### Results :
There are several Jupyter notebooks that have already run the models and methods:
- */Data analysis.ipynb*: Data exploration 
- */model + prior*: Different Models with different Priors : Model ( Gaussian , Student's), Prior ( Gaussian ,Laplace , Expert ).

