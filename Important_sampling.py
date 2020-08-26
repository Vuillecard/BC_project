import time
import torch 
import pandas as pd
import numpy as np

class Important_sampling:
    """
    Important sampling with unormalized density using a weighted method
    the estimate is consistent and biased (in generale the biased is small)
    
    Attributes:
        model (object): Obeject from the Bayesian_Framework class, that define the model and the data
    """
    
    def __init__(self, model):
        self.model = model
        
    def compute_algo(self , proposal , num_sample ):
        """
        Algorithm of the important sampling method
        Strong condition on the proposal :
        if g(x)=0 then f(x)=0
        
        Args:
            proposal (function): Proposal density function supported scipy.stats
            Num_sample (int): The size of the sample 
            
        Returns:
            the weight w and the sample from the proposal theta
        """
        self.num_sample = num_sample
        sample_proposal = proposal.rvs(size= num_sample)
        f_tild = self.model.compute_log_posterior()
        f_tild_evaluate = f_tild(torch.as_tensor(sample_proposal).t()).numpy()
        proposal_evaluate = proposal.pdf(sample_proposal)
        w = f_tild_evaluate/proposal_evaluate
        self.w = w
        self.sample_proposal = sample_proposal
        return w , sample_proposal
    
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
    
    def compute_mode_algo(self, step_size = 0.1,max_iteration = 500 , line_search = False , stop_criterion = 0.1 ):
        """
        Find the mode of the model's posterior
        
        Args:
            step_size (double): step of the optimization part
            max_iteration (int): the maximal number of optimization loop to compute
            line_search (bool): true to use a backtracking line search in the optimization step
            stop_criterion (double): it define the epsilon such that f_n - f_n+1 < epsilon then stop 
            
        Returns:
            The optimal theta that maximise the posterior 
            
        Note:
            If line_search is true then the step_size is automatically selected, thus step_size is never used.
        """
        start_time = time.time()
        phi = self.model.compute_log_posterior()
        target = lambda x : -phi(x)
        theta = 10*torch.ones(self.model.d,1)
        loss_prev = target(theta)+10
        for i in range(max_iteration):

            if theta.grad is not None :
                theta.grad.detach_()
                theta.grad.zero_()
            theta.requires_grad_(True)
            # generate sample from standard gaussian 

            loss = target(theta)
            
            if loss_prev-loss<stop_criterion :
                print( " Mode find out ")
                break
            loss_prev = loss 
            loss.backward()
            #Optimization step
            print(loss)
            #print(torch.norm(theta.grad))
            with torch.no_grad():
                if line_search :
                    theta = self.backtracking_line_search(theta.grad,target , theta)
                else :
                    theta = theta - step_size*theta.grad

        # stock the results
        print(" It takes :", round(time.time()-start_time,2)," s to find the mode ")
        return theta.requires_grad_(False)
        
    def compute_statistic(self,burn_in= 0):
        """
        Compute a few statistic based on the sample 
        """
        self.burn_in = burn_in
        mean = np.sum(np.reshape(self.w,(self.num_sample,1))*self.sample_proposal,axis= 0)/np.sum(self.w)
        var = np.sum(np.reshape(self.w,(self.num_sample,1))*(self.sample_proposal**2),axis= 0)/np.sum(self.w) - mean**2
        self.posterior_mean = mean
        self.posterior_std  = np.sqrt(var)
        tmp_2 = self.posterior_mean + 2*self.posterior_std 
        tmp_1 = self.posterior_mean - 2*self.posterior_std
        self.posterior_IC   = np.c_[tmp_1 ,tmp_2] # d*2 array  
        
        
    def summary(self):
        """
        Print the results of the approximation  
        """
        table = pd.DataFrame()
       
        table["features"]= self.model.name 
        table["Coeff"]= [round(self.posterior_mean[i],2) for i in range(self.model.d)]
        table["std"] = [ round(self.posterior_std[i],2) for i in range(self.model.d)]
        table["CI"]= ["["+"{:.3f}".format(self.posterior_IC[i,0])+", "+"{:.3f}".format(self.posterior_IC[i,1])+"]"  for i in range(self.model.d)]
        print(table)