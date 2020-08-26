import numpy as np
import tqdm
import pandas as pd
import time
import torch

class GVA :
    """
    Gaussian variational approximation : full-rank
    
    Attributes:
        model (object): A Bayesian model from the bayesian framework object 
        sample_size (int): The size of the sample that it generates
    """
    
    def __init__(self,Bayesian_model , sample_size = 10000):
        self.model = Bayesian_model
        self.sample_size = sample_size
        
        
    def ELBO(self , mu , L , eta):
        """
        The Evidence-Lower BOund function
        
        Args:
            mu (double): define the mean of the posterior
            L (double): Matrix that describe the covariance of the posterior
            eta (double): dxN Sample from a multivariate standard normal
            
        Returns:
            The ELBO function
        """
        
        phi = self.model.compute_log_posterior()
        tmp = torch.matmul(torch.exp(L),eta.t()) + torch.reshape(mu, (self.model.d,1))
        results = -phi(tmp)
        
        return -( -results.mean() + (self.model.d/2)*torch.log(np.pi*2*torch.exp(torch.ones(1))) + L.trace() )
    
    def backtracking_line_search(self, grad, target, theta_old, beta = 0.8, alpha = 0.5 ):
        """
        Algorithm that select the best step_size in the Gradient Descent
        
        Args:
            grad (double): gradient of the targeet wrt to theta_old
            target (double): the function that is optimized
            theta_old (double): previous variable of the optimization step
            beta (double): in ]0,1[, define the reduction rate of the step_size closer to 1 the slower it deacrease
            alpha (double): in ]0,1/2[ parameter of the algorithm 
        Returns:
            the best variable that minimize the target function  
        """
        lambda_ = 1
        #decrease lambda_ such that value of sigma > 0
        while np.isnan(target(theta_old - lambda_*grad)) :
            lambda_ = 0.8*lambda_
            
        # decrease lambda_ until a good convergence rate 
        while target(theta_old - lambda_*grad) > (target(theta_old) - lambda_*alpha*(np.linalg.norm(grad,2)**2)) :
            lambda_ = beta*lambda_
    
        return theta_old - lambda_*grad
    
    def compute_GVA_algo(self,init_mu,init_L, step_size = 0.01,num_samples = 1000,max_iteration = 300 , stop_crit = 0.5 , show = False , line_search = False ):
        """
        Main algorithm that maximize the ELBO via a stochatstic gradient descent (SGD)
        
        Args:
            init_mu (double): parameter that describe the mean of the posterior
            init_L (double): parameter that describe the covariance of the posterior
            step_size (double): use for the optimization step
            num_samples (int): The sample size of the auxiliary distribution
            max_iteration (int): The maximum number of iteration to do per optimization step 
            stop_criterion (double): stopping criterion that stop the optimization when the norm of the gradient is small enough
            show (bool): A display argument that print the results during the optimization 
            line_search (bool):True to use a backtracking line search that optimize the step_size of the SGD
        
        Returns:
            Nothing but stock the sample that has been generated with a few statistic 
        """
        start_time = time.time()
        step_size_ = step_size*torch.ones(self.model.d)
        phi = self.model.compute_log_posterior()
        #initialise the variable 
        mu = init_mu
        L = init_L
        loss_prev = 0
        for i in tqdm.tqdm(range(max_iteration)):
            
            # initialise the gradient 
            if mu.grad is not None :
                mu.grad.detach_()
                mu.grad.zero_()
            if L.grad is not None :
                L.grad.detach_()
                L.grad.zero_()
            mu.requires_grad_(True)
            L.requires_grad_(True)
            
            # computing the -ELBO function
            eta = torch.empty(num_samples,self.model.d).normal_()
            loss = self.ELBO(mu ,L ,eta)
            
            #computing the gradient 
            loss.backward()
            
            # Print the convergence steps 
            if show and i%10==0 :
                print(" the loss is :" , loss)
                print(" the grad is :" , mu.grad) 
                print(" norm of the gradient ", torch.norm(mu.grad))
            
            if torch.norm(mu.grad) < stop_crit :
                print( " Norm gradient < 0.5 stop ")
                break 
                
            #Optimization step
            with torch.no_grad():
                
                if line_search :
                    mu = self.backtracking_line_search(mu.grad, lambda x : self.ELBO(x ,L,eta ) , mu )
                    L = self.backtracking_line_search(L.grad, lambda x : self.ELBO(mu ,x,eta ) , L )
                    
                else :
                    #mu = mu - step_size_*mu.grad
                    mu.sub_(step_size_*mu.grad)
                    #L = L - step_size*L.grad
                    L.sub_(step_size*L.grad)
                    
                if show and i%10==0 :
                    print(mu)
                    
        # time recording :
        print(" GVA method takes :", round(time.time()-start_time,2)," s to run ")
        
        # stock the results
        mu.requires_grad_(False)
        L.requires_grad_(False)
        self.mu_post = mu
        self.L_post = L
        
        # Samples and stat computation 
        self.generate_sample()
        self.compute_statistic()
        self.summary()
        
    
    def generate_sample(self):
        """
        Generate a sample base on the GVA results 
        """
        eta = torch.empty(self.sample_size,self.model.d).normal_()
        samples = torch.matmul(torch.exp(self.L_post),eta.t()) + torch.reshape(self.mu_post, (self.model.d,1))
        self.samples = samples.t().detach().numpy()
        
    def compute_statistic(self,burn_in= 0):
        """
        Compute a few statistic based on the sample 
        """
        self.burn_in = burn_in
        self.posterior_mean = np.mean(self.samples[burn_in:], axis = 0)
        self.posterior_std  = np.std(self.samples[burn_in:], axis = 0)
        tmp_2 = self.posterior_mean + 2*self.posterior_std 
        tmp_1 = self.posterior_mean - 2*self.posterior_std
        self.posterior_IC   = np.c_[tmp_1 ,tmp_2] # d*2 array  
        
        #quantile :
        self.quantile_pourcent = np.array([2.5,25,50,75,97.5])
        sort_sample = np.sort(self.samples[burn_in:],axis=0)
        n_sort_sample = np.shape(sort_sample)[0]
        indice_quantile = (n_sort_sample*self.quantile_pourcent/100).astype('int32') 
        self.quantile = sort_sample[indice_quantile,:] # 5*d array 
        
    def summary(self):
        """
        Print the results of the approximation  
        """
        table = pd.DataFrame()
       
        table["features"]= self.model.name 
        table["Coeff"]= [round(self.posterior_mean[i],2) for i in range(self.model.d)]
        table["std"] = [ round(self.posterior_std[i],2) for i in range(self.model.d)]
        table["CI"]= ["["+"{:.3f}".format(self.posterior_IC[i,0])+", "+"{:.3f}".format(self.posterior_IC[i,1])+"]"  for i in range(self.model.d)]
        
        for i in range(5):
            table[str(self.quantile_pourcent[i])] = [ round(self.quantile[i,j],2) for j in range(self.model.d)]
        print(table)