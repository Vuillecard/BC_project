import numpy as np
import tqdm
import pandas as pd
import scipy.stats as st

class Metropolis_hastings :
    """
    The Metropolis hastings class
    
    Attributes:
        model (object): Model from the bayesian framework 
        init (double): Initial variable estimation 
        step_size (double): Step size that define the variance of the gaussian step
        num_sample (int): the sample size
        
    Note :
        If the set_size is to small then the algorithm do not discover the all space of the posterior, Thus can miss part of the distribution 
        
    """
    def __init__(self,Bayesian_model,init,step_size = 0.01, num_samples = 10000, decimation = 1 , mode = False):
        self.model = Bayesian_model
        self.init = init
        self.step_size= step_size
        self.num_samples = num_samples
        self.decimation = decimation
        self.mode = mode
        
        self.compute_Metropolis_hastings_algo()
    
    def compute_Metropolis_hastings_algo(self):
        """
        Main function that compute the metropolos hasting algorithm 
        """
        
        log_posterior_function = self.model.compute_log_posterior()
        
        samples = np.zeros( [self.num_samples, self.model.d] )
        record_acceptance = np.zeros( [self.num_samples] )
        current = self.init # order is intercepte , predictor, sigma 

        for k in tqdm.tqdm(range(self.num_samples)):
            for dummy in range(self.decimation):
                proposal = current + self.step_size * np.random.randn(self.model.d)
                ratio_MH = np.exp( log_posterior_function(proposal) - log_posterior_function(current) )

                uniform_RV = np.random.random()
                if uniform_RV < ratio_MH:
                    current = proposal

            samples[k,:] = current
            record_acceptance[k] = (uniform_RV < ratio_MH)

        print("Acceptance rate : ", np.mean(record_acceptance) )
        self.samples = samples
        self.acceptance_rate = np.mean(record_acceptance)
        
        
    def compute_statistic(self,burn_in= 0):
        """
        Compute some statistic based on the sample of the algorithm
        """
        self.burn_in = burn_in
        if self.mode :
            self.posterior_mean = np.asarray((st.mode(self.samples[burn_in:], axis = 0)).mode[0])
            print(self.posterior_mean)
        else : 
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
        Print the results of the sampling algorithm 
        """
        
        table = pd.DataFrame()
       
        table["features"]= self.model.name 
        table["Coeff"]= [round(self.posterior_mean[i],2) for i in range(self.model.d)]
        table["std"] = [ round(self.posterior_std[i],2) for i in range(self.model.d)]
        table["CI"]= ["["+"{:.3f}".format(self.posterior_IC[i,0])+", "+"{:.3f}".format(self.posterior_IC[i,1])+"]"  for i in range(self.model.d)]
        
        for i in range(5):
            table[str(self.quantile_pourcent[i])] = [ round(self.quantile[i,j],2) for j in range(self.model.d)]
        print(table)
            
        """
        print("=============================summary==============================")
        for ind,name in enumerate(self.model.name) :
            print(name + " :")
            print("Mean\tstd\t  95% CI ")
            message_1 ="{:.3f}".format(self.posterior_mean[ind])
            message_2 ="\t"+"{:.3f}".format(self.posterior_std[ind])
            message_3 ="["+"{:.3f}".format(self.posterior_IC[ind,0])+", "+"{:.3f}".format(self.posterior_IC[ind,1])+"]"
            print(message_1+message_2+message_3)
            print("")
            print("poseterior quantile :")
            message = ""
            message_aux = ""
            inter = ""
            for i in range(5):
                message_aux += "  "+str(self.quantile_pourcent[i])+"\t"
                inter += "|------|"
                message += "  "+"{:.3f}".format(self.quantile[i,ind])+"\t"
            print(message_aux)
            print(inter)
            print(message)
            print("==================================================================")
         """
                               