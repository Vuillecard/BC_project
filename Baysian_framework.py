
import numpy as np
import torch


class Bayesian_framework :
    """
    This class defines the data and the model of the bayesian framework
    
    Attributes:
        d (int): The total number of parameter of the model
        p (int): The number of slopes predictor
        n (int): The number observation 
        name (string): Name of the parameter
        torch_implementation (bool): true if the model is implemented usig pytorch
        transformation (bool): true if the target train is modify with log 
        transformation_test (bool): true if the target test is modify with log
        beta_hat : basic statistical inference 
        X : predictors
        Y : target
    """
    def __init__(self,p,Name,X,Y, transformation_test = False , transformation_train = False ,ratio=0.8,torch_imp = False ):
        
        self.d = len(Name)
        self.p = p # number of predictor without intercept
        self.n = np.shape(X)[0] # number of observation
        self.name = Name # Name of the variable 
        self.torch_implementation = torch_imp
        self.transformation = transformation_train
        self.transformation_test = transformation_test
        self.data_normalization(X,Y,ratio)
        self.beta_hat = np.linalg.solve(self.X_train.T@self.X_train,self.X_train.T@self.Y_train)
        self.X = X
        self.Y = Y
       
        
    
    
    def data_normalization(self,X,Y,ratio):
        """
        Initialise the data and normalize it
        
        Args:
            X : predictors
            Y : target
            ratio (double): [0,1] Part of the training and test set default 80% training 
        """
        np.random.seed(123)
        n_observation = np.shape(X)[0]
        n_train = int(n_observation*ratio)
        permuted_indice = np.random.permutation(n_observation)
        X_permuted = np.copy(X[permuted_indice])
        Y_permuted = np.copy(Y[permuted_indice])
        self.n_train = n_train
        self.n_test = n_observation - n_train
        mean =np.mean(X_permuted[:n_train],axis=0)
        std = np.std(X[:n_train],axis=0)
        X_permuted = (X_permuted-mean)/std
        X_one = np.c_[np.ones(n_observation),X_permuted]
        
        
        if  self.torch_implementation : 
            
            self.X_train = torch.as_tensor(X_one[:n_train,:])
            self.X_test = torch.as_tensor(X_one[n_train:,:])
            self.Y_train = torch.as_tensor(Y_permuted[:n_train])
            self.Y_test = torch.as_tensor(Y_permuted[n_train:])
        else : 
            
            self.X_train = X_one[:n_train,:]
            self.X_test = X_one[n_train:,:]
            self.Y_train = Y_permuted[:n_train]
            self.Y_test = Y_permuted[n_train:]
        
        
    def set_log_joint_prior(self,joint_prior):
        self.log_joint_prior = joint_prior()
    
    def set_log_likelihood(self,likelihood):
        f = lambda y,x : likelihood(y,x)
        self.log_likelihood = f
    
    def compute_log_likelihood(self,variable):
        ans=0
        for ind_y ,y in enumerate(self.Y_train):
                log_likelihood_datapoint = self.log_likelihood(y,self.X_train[ind_y,:])
                ans = ans+ log_likelihood_datapoint( variable ) 
        return ans
    
    def compute_log_likelihood_test(self,variable):
        ans=0
        for ind_y ,y in enumerate(self.Y_test):
                log_likelihood_datapoint = self.log_likelihood(y,self.X_test[ind_y,:])
                ans = ans+ log_likelihood_datapoint( variable ) 
        return ans 
        
    def compute_log_posterior(self):
        
        def anonymous_function( variable ):
            ans = self.log_joint_prior( variable )

            for ind_y ,y in enumerate(self.Y_train):
                log_likelihood_datapoint = self.log_likelihood(y,self.X_train[ind_y,:])
                ans = ans+ log_likelihood_datapoint( variable ) 

            return ans
            
        return anonymous_function