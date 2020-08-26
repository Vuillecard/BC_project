import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
import torch 

class Vizualisation_tool:
    """
    This class brings together all function use to vizualise the result and also compute some statistic of the model
    
    Attributes:
        model (object): Model object from the bayesian framwork class
        sample_alg (object): The method use to compute an approximation of the posterior
        torch_imp (bool): True if the implementation of the model use Pytorch 
    """
    
    def __init__(self,model, sample_alg ):
        # seaborn plot 
        sns.set()
        self.model = model 
        self.sample_alg = sample_alg
        self.torch_imp = model.torch_implementation
        
        
    def autocorrelation(self,time_series, maxRange):
        """
        estimate the autocorrelation, correctly normalized, for all the lags in the delta array
        """
        l = len(time_series)

        ans = np.zeros(2*maxRange+1)
        delta = np.arange(-maxRange,maxRange+1,1)

        for k in range(2*maxRange+1):
            v0 = time_series[maxRange            : l - maxRange           ]
            v1 = time_series[maxRange - delta[k] : l - maxRange - delta[k]]

            m0 = np.mean(v0)
            m1 = np.mean(v1)
            cov = np.sum( (v0-m0) * (v1-m1) / len(v0) )
            var0 = np.sum( (v0-m0)**2 / len(v0) )
            var1 = np.sum( (v1-m1)**2 / len(v0) )
            corr = cov / (var0 * var1)**0.5

            ans[k] = corr

        return delta, ans
    

    def showAutocorrelation(self,samples, delta = None):
        """
        Plots the autocorrelations
        """
        if delta == None:
            delta = np.int( len(self.sample_alg.samples) / 6 )
        _, trueCorrelation = self.autocorrelation(samples, delta )

        #plt.plot(np.arange(-len(reduced),len(reduced)-1),correlation / correlation.max(), 'o')
        plt.plot(np.arange(-delta,delta+1), trueCorrelation, 'o')
        plt.ylim([-1,1])
        #plt.show()
    
    def trace_visualization(self):
        """
        Plots the traces of the markov chain from the metropolis hastings
        """
        for i in range(self.model.d):
            plt.figure(figsize=(7,7))
            plt.plot(self.sample_alg.samples[:,i])
            plt.title("Autocorrelation of "+self.model.name[i])
            plt.show()
    
    def autocorrelation_visualization(self,delta=1000):
        for i in range(self.model.d):
            plt.figure(figsize=(7,7))
            self.showAutocorrelation(self.sample_alg.samples[ self.sample_alg.burn_in:,i],delta) 
            plt.title("Autocorrelation of "+self.model.name[i])
            plt.show()
    
    def marginal_visualization(self):
        """
        Plot the marginal posterior distribution
        """
        for i in range(self.model.d):
            plt.figure()
            sns.kdeplot(self.sample_alg.samples[self.sample_alg.burn_in:,i])
            plt.axvline(x=self.sample_alg.posterior_mean[i],color='red')
            plt.title("Approximate marginal density of "+self.model.name[i])
            plt.show()
  
    def qq_plot(self,distribution):
        """
        Plot the qq plot of the standardize residual compare to a gicen distribution
        
        Args:
            distribution (scipy.stats): Distribution to compare
            
        Returns:
            Plots of the qq plot
        """
        # stats.t(df=posterior_mean_approximation[-1])
        plt.figure()
        stats.probplot(self.residual_standardize(),dist=distribution, plot=plt);
        plt.show()
        
    def transformation_exp(self,x) :
        """
        Function that transform the output variable, when the outputs are in log
        """
        if self.model.transformation :
            if self.torch_imp :
                y = np.exp(x.numpy())
            else :
                y = np.exp(x)
        else:
            if self.torch_imp :
                y = x.numpy()
            else :
                y = x
                
        return y
    
    def y_hat(self):
        """
        Compute the prediction based on the posterior mean as coefficient of the linear regression 
        """
        coef = self.sample_alg.posterior_mean[:self.model.p+1]
        if self.model.transformation :
            if self.torch_imp :
                y_hat = np.exp((self.model.X_train).numpy()@coef)
            else :
                y_hat = np.exp(self.model.X_train@coef)
        else:
            if self.torch_imp:
                y_hat = (self.model.X_train).numpy()@coef
            else :
                y_hat = self.model.X_train@coef
            
        return y_hat
    
   
        
    def y_bar(self):
        """
        Compute the mean of the target
        """
        y = np.mean(self.transformation_exp(self.model.Y_train))
        return y
    
    def residual(self):
        """
        Compute the residuals
        """
        y = self.transformation_exp(self.model.Y_train)
        return y - self.y_hat()
    
    def residual_standardize(self):
        """
        Compute the standardize residuals
        """
        ind_sigma = np.where(np.asarray(self.model.name) == "sigma")[0][0]
        return self.residual()/self.sample_alg.posterior_mean[ind_sigma]
        
        
    def SSE(self):
        
        return np.sum(np.power(self.residual(),2))
    
    def SSR(self):
        
        tmp = self.y_hat() - self.y_bar()
        return np.sum(np.power(tmp,2))
    
    def SSTO(self):
        
        y = self.transformation_exp(self.model.Y_train)
        tmp = y - self.y_bar()
        return np.sum(np.power(tmp,2))
    
    def MSE(self):
        
        return self.SSE()/(self.model.n_train -self.model.p - 1)
    
    def MSE_test(self):
        
        coef = self.sample_alg.posterior_mean[:self.model.p+1]
        
        if self.model.transformation_test :
            if self.torch_imp :
                y_hat = np.exp((self.model.X_test).numpy()@coef)
                y = np.exp(self.model.Y_test.numpy())
            else :
                y_hat = np.exp(self.model.X_test@coef)
                y = np.exp(self.model.Y_test)
        else:
            if self.torch_imp:
                y_hat = (self.model.X_test).numpy()@coef
                y = self.model.Y_test.numpy()
            else :
                y_hat = self.model.X_test@coef
                y = self.model.Y_test
                
        SSE = np.sum(np.power(y-y_hat,2))
        return SSE/(self.model.n_test -self.model.p - 1)
    
    def MSR(self):
        
        return self.SSR()/self.model.p
    
    def PVE(self):
        """
        Proportion of explained variation always between 0 and 1
        1 indicate a closer fit to the data
        """
        return self.SSR()/self.SSTO()
        
    def F_test(self):
        
        F = self.MSR()/self.MSE()
        p_value = stats.f.cdf(F, self.model.p , self.model.n_train - self.model.p - 1)
        return F , 1 - p_value
        
        
    def BIC(self):
        
        d = self.model.p + 1
        if self.torch_imp :
            mean_variable = torch.as_tensor(self.sample_alg.posterior_mean)
        else :
            mean_variable = self.sample_alg.posterior_mean
            
        log_L = np.asarray(self.model.compute_log_likelihood(mean_variable))
        return d*np.log(self.model.n_train) - 2*log_L
    
    def AIC(self):
        
        d = self.model.p + 1
        if self.torch_imp :
            mean_variable = torch.as_tensor(self.sample_alg.posterior_mean)
        else :
            mean_variable = self.sample_alg.posterior_mean
            
        log_L = np.asarray(self.model.compute_log_likelihood(mean_variable))
        return 2*d - 2*log_L
    
    def AICc(self):
        
        d = self.model.p + 1 
        return self.AIC() + (2*d*(d+1))/(self.model.n_train-d-1)
    
    
    def fidelity(self ):
        """
        Print all the statistic of the model
        """
        
        print("============ Fidelity summary ==================")
        F , p_value = self.F_test()
        print(" the mean square error is : ",self.MSE())
        print(" MSE test is : ",self.MSE_test())
        print(" PVE is  : ", self.PVE())
        
        print(" F   is  : ", F)
        print(" p value : ", p_value)
        print(" AIC is  : ", self.AIC())
        print(" BIC is  : ", self.BIC())
        print(" AICc is : ", self.AICc())
        
        
    def diagnostics(self , student = False ):
        """
        Plot diagnostics of the model 
        """
        # residual vs fitted value 
        plt.figure()
        plt.scatter(self.y_hat(),self.residual())
        plt.xlabel('Fitted value')
        plt.ylabel('Residuals')
        
        #histogram of residual 
        plt.figure()
        if student :
            sns.distplot(self.residual_standardize(),fit = stats.t );
        else :
            sns.distplot(self.residual_standardize(),fit = norm);
        
        #qq plot
        plt.figure()
        if student :
            self.qq_plot(stats.t(df=self.sample_alg.posterior_mean[-1]))
        else :
            self.qq_plot("norm")
        
        # residual vs predictor 
        for i in range(self.model.p):
            plt.figure()
            plt.scatter(self.model.X_train[:,i+1],self.residual())
            plt.xlabel(self.model.name[i+1])
            plt.ylabel('Residuals')
            
        # cluster information :
        """
        abs_residual = np.abs(self.residual())
        sort_abs_residual = np.sort(abs_residual)
        ind_75_pctl = int(np.shape(sort_abs_residual)[0]*0.75)
        ind_90_pctl = int(np.shape(sort_abs_residual)[0]*0.90)
        value_75_pctl = sort_abs_residual[ind_75_pctl]
        value_90_pctl = sort_abs_residual[ind_90_pctl]
        indice_0_75_pctl = np.where(abs_residual<=value_75_pctl)[0]
        indice_90_100_pctl = np.where(abs_residual>value_90_pctl )[0]
        indice_75_90_pctl = []
        for i in range(np.shape(sort_abs_residual)[0]):
            if i not in indice_0_75_pctl:
                if i not in indice_90_100_pctl:
                    indice_75_90_pctl.append(i)
        plt.figure()
        plt.scatter(self.model.X_test[indice_0_75_pctl,6],self.model.X_test[indice_0_75_pctl,5],label='0-75 pctl')
        plt.scatter(self.model.X_test[indice_75_90_pctl,6],self.model.X_test[indice_75_90_pctl,5],label='75-90 pctl')
        plt.scatter(self.model.X_test[indice_90_100_pctl,6],self.model.X_test[indice_90_100_pctl,5],label='90-100 pctl')
        plt.legend()
        plt.xlabel(self.model.name[6])
        plt.ylabel(self.model.name[5])
        plt.show()
        """ 
        
        
              
def F_test_reduce_model(tool_model_full , tool_model_reduce ):
    
    """
    df1 is the difference of parameter and df2 is n-p-1 where p is the number of predictor of the full model
    If p value is small we can reject the null
    """
    df1 = tool_model_full.model.p - tool_model_reduce.model.p
    df2 = tool_model_full.model.n_train - tool_model_full.model.p - 1
    
    F = ((tool_model_full.SSR() - tool_model_reduce.SSR())/df1)/tool_model_full.MSE()
    p_value = stats.f.cdf(F, df1 , df2 )
    return F , 1 - p_value  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       
        
        
        
        
        
        
        
        
        
