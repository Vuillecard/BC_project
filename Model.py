import numpy as np
import scipy as scp # optimization, probability densities and cumulative

"""
This file define all the model that we use :

    A : Gaussian model with gaussina prior 
    B : Gaussina model with laplace prior 
    C : Student model with Gaussian prior
    D : Student model with Laplace prior 
    E : Gaussian model with expert prior 

"""
##############################################################################################################
############################## A : Gaussian model with gaussina prior ########################################
##############################################################################################################

class Gaussian_model_np :
    """ 
    the variable need to be like [ slope , slope_quad , sigma ]
    """
    def __init__(self,model,nb_quad_term = 0):
        self.model = model
        self.d = (self.model.p+1)
        self.range_sigma = np.sqrt(np.var(model.Y)/(self.model.p+1))
        self.nb_quad_term = nb_quad_term
        
    def log_gaussain_prior_B(self):
        d= self.d - self.nb_quad_term
        m_0= np.zeros( self.d - self.nb_quad_term )
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
    
    def log_gaussain_prior_B_quadratic(self):
        d= self.nb_quad_term
        m_0= np.zeros( self.nb_quad_term )
        # here we would like that the quadratic term are close to zero
        sigma_b_0= self.range_sigma/(8)
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
    
    # k_0 =4 for gaussian and 2 if log
    def log_gamma_prior_sigma(self ,k_0=2, theta_0=2):
        f = lambda sigma : -k_0*np.log(theta_0)-np.real(scp.special.loggamma(k_0)) + (k_0-1)*np.log(sigma) - sigma/theta_0
        return f
    
    def log_joint_prior(self):
        
        log_prior_beta = self.log_gaussain_prior_B()
        log_prior_beta_quad = self.log_gaussain_prior_B_quadratic()
        log_prior_sigma = self.log_gamma_prior_sigma()
        
        if self.nb_quad_term > 0 :
            ans = lambda variable : log_prior_beta(variable[0:self.d-self.nb_quad_term]) + log_prior_beta_quad(variable[self.d-self.nb_quad_term :self.d]) + log_prior_sigma(variable[-1])
        else :
            ans = lambda variable : log_prior_beta(variable[0:self.d])  + log_prior_sigma(variable[-1])
            
        return ans
    
    def log_likelihood(self,y,x):
        ans = lambda variable : np.real(
                                    - 0.5* np.log(2 * np.pi) 
                                    - 0.5*np.log( variable[-1]**2 ) 
                                    - 0.5 *((y-np.dot(x,variable[:self.d]))**2) / (variable[-1])**2)
        return ans
##############################################################################################################
############################## B : Gaussina model with laplace prior  ########################################
##############################################################################################################    
    
class Gaussian_model_laplace_prior :
    """ 
    the variable need to be like [ slope , slope_quad , sigma ]
    """
    def __init__(self,model , nb_quad_term = 0 , coeff = 4):
        self.model = model
        self.d = (self.model.p+1)
        self.range_sigma = np.sqrt(np.var(model.Y)/(self.model.p+1))
        self.nb_quad_term = nb_quad_term
        self.coeff = coeff
        
    def log_gaussain_prior_intercept(self):
        d= 1
        m_0= np.zeros( 1 )
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
       
        
    def log_laplace_prior_B(self):
        d= self.d- self.nb_quad_term - 1
        m_0= np.zeros( self.d- self.nb_quad_term - 1)
        b= np.sqrt(self.range_sigma/2)/self.coeff
        
        f = lambda x : - d*np.log(2*b) -(1/b)*(np.sum(np.abs(x-m_0)))
        return f
    
    def log_laplace_prior_quad_B(self):
        d= self.nb_quad_term 
        m_0= np.zeros( self.nb_quad_term )
        b= np.sqrt(self.range_sigma/2)/(self.coeff*2)
        
        f = lambda x : - d*np.log(2*b) -(1/b)*(np.sum(np.abs(x-m_0)))
        return f
    
    # k_0 =4 for gaussian 
    def log_gamma_prior_sigma(self ,k_0=4, theta_0=2):
        f = lambda sigma : -k_0*np.log(theta_0)-np.real(scp.special.loggamma(k_0)) + (k_0-1)*np.log(sigma) - sigma/theta_0
        return f
    
    def log_joint_prior(self):
        log_prior_intercept = self.log_gaussain_prior_intercept()
        log_prior_beta = self.log_laplace_prior_B()
        log_prior_beta_quad = self.log_laplace_prior_quad_B()
        log_prior_sigma = self.log_gamma_prior_sigma()
        
       
        if self.nb_quad_term > 0 :
            ans = lambda variable :log_prior_intercept(variable[0]) + log_prior_beta(variable[1:self.d-self.nb_quad_term]) + log_prior_beta_quad(variable[self.d-self.nb_quad_term :self.d]) + log_prior_sigma(variable[-1])
        else :
            ans = lambda variable : log_prior_intercept(variable[0]) + log_prior_beta(variable[1:self.d])  + log_prior_sigma(variable[-1])
            
        return ans
    
    def log_likelihood(self,y,x):
        ans = lambda variable : np.real(
                                    - 0.5* np.log(2 * np.pi) 
                                    - 0.5*np.log( variable[-1]**2 ) 
                                    - 0.5 *((y-np.dot(x,variable[:self.d]))**2) / (variable[-1])**2)
        return ans
    
##############################################################################################################
##############################  C : Student model with Gaussian prior ########################################
##############################################################################################################
        
class Student_model_np :
    """ 
    the variable need to be like [ slope , slope_quad , sigma , df ]
    """
    def __init__(self,model,nb_quad_term = 0):
        self.model = model
        self.d = (self.model.p+1)
        self.range_sigma = np.sqrt(np.var(model.Y)/(self.model.p+1))
        self.nb_quad_term = nb_quad_term
        
    def log_gaussain_prior_B(self):
        d= self.d - self.nb_quad_term
        m_0= np.zeros( self.d - self.nb_quad_term )
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
    
    def log_gaussain_prior_B_quadratic(self):
        d= self.nb_quad_term
        m_0= np.zeros( self.nb_quad_term )
        # here we would like that the quadratic term are close to zero
        sigma_b_0= self.range_sigma/(2*(2**2))
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
    
    def log_gamma_prior_nu(self, a_0 = 2, b_0 = 2):
        ans = lambda nu: -a_0 * np.log(b_0) - np.real(scp.special.loggamma(a_0)) + (a_0 - 1) * np.log( nu ) - nu/b_0
        return ans  
    
    # k_0 =4 for gaussian and 2 if log
    def log_gamma_prior_sigma(self ,k_0=4, theta_0=2):
        f = lambda sigma : -k_0*np.log(theta_0)-np.real(scp.special.loggamma(k_0)) + (k_0-1)*np.log(sigma) - sigma/theta_0
        return f
    
    def log_likelihood(self,y,x):
        ans = lambda variable : np.real(
                                    - np.log(variable[-2])                              # changing sigma changes the density
                                    + scp.special.loggamma( (variable[-1]+1)/2 )            # normal term of the student density
                                    - 0.5 *np.log( variable[-1] * np.pi )                  # normal term of the student density
                                    - scp.special.loggamma( variable[-1]/2 )                # normal term of the student density
                                    - ((variable[-1]+1)/2) * np.log(1 + ((y-np.dot(x,variable[0:self.d]))**2) /((variable[-2]**2)*variable[-1]))   )
    
        return ans
    
    
    def log_joint_prior(self):
        
        log_prior_beta = self.log_gaussain_prior_B()
        log_prior_beta_quad = self.log_gaussain_prior_B_quadratic()
        log_prior_sigma = self.log_gamma_prior_sigma()
        log_prior_nu = self.log_gamma_prior_nu()
        
        if self.nb_quad_term > 0 :
            ans = lambda variable : log_prior_beta(variable[0:self.d-self.nb_quad_term]) + log_prior_beta_quad(variable[self.d-self.nb_quad_term :self.d]) + log_prior_sigma(variable[-1])
        else :
            ans = lambda variable : log_prior_beta(variable[0:self.d])  + log_prior_sigma(variable[-2]) + log_prior_nu(variable[-1])
            
        return ans
##############################################################################################################
##############################  D : Student model with Laplace prior  ########################################
##############################################################################################################    
    
class Student_model_laplace_prior :
    """ 
    the variable need to be like [ slope , slope_quad , sigma , df ]
    """
    def __init__(self,model,nb_quad_term = 0):
        self.model = model
        self.d = (self.model.p+1)
        self.range_sigma = np.sqrt(np.var(model.Y)/(self.model.p+1))
        self.nb_quad_term = nb_quad_term
    
    def log_gaussain_prior_intercept(self):
        d= 1
        m_0= np.zeros( 1 )
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
   
    
    def log_laplace_prior_B(self):
        d= self.d-1
        m_0= np.zeros( self.d-1)
        b= np.sqrt(self.range_sigma/2)/6
        
        f = lambda x : - d*np.log(2*b) -(1/b)*(np.sum(np.abs(x-m_0)))
        return f
    
    def log_gamma_prior_nu(self, a_0 = 2, b_0 = 2):
        ans = lambda nu: -a_0 * np.log(b_0) - np.real(scp.special.loggamma(a_0)) + (a_0 - 1) * np.log( nu ) - nu/b_0
        return ans  
    
    # k_0 =4 for gaussian and 2 if log
    def log_gamma_prior_sigma(self ,k_0=4, theta_0=2):
        f = lambda sigma : -k_0*np.log(theta_0)-np.real(scp.special.loggamma(k_0)) + (k_0-1)*np.log(sigma) - sigma/theta_0
        return f
    
    def log_likelihood(self,y,x):
        ans = lambda variable : np.real(
                                    - np.log(variable[-2])                              # changing sigma changes the density
                                    + scp.special.loggamma( (variable[-1]+1)/2 )            # normal term of the student density
                                    - 0.5 *np.log( variable[-1] * np.pi )                  # normal term of the student density
                                    - scp.special.loggamma( variable[-1]/2 )                # normal term of the student density
                                    - ((variable[-1]+1)/2) * np.log(1 + ((y-np.dot(x,variable[0:self.d]))**2) /((variable[-2]**2)*variable[-1]))   )
    
        return ans
    
    
    def log_joint_prior(self):
        log_prior_intercept = self.log_gaussain_prior_intercept()
        log_prior_beta = self.log_laplace_prior_B()
        log_prior_sigma = self.log_gamma_prior_sigma()
        log_prior_nu = self.log_gamma_prior_nu()
        
        ans = lambda variable : log_prior_intercept(variable[0])+ log_prior_beta(variable[1:self.d])  + log_prior_sigma(variable[-2]) + log_prior_nu(variable[-1])
            
        return ans

##############################################################################################################
############################## E : Gaussian model with expert prior ##########################################
##############################################################################################################

class Gaussian_model_expert_prior :
    """ 
    the variable need to be like [ slope , slope_quad , sigma ]
    """
    def __init__(self,model,nb_quad_term = 0 , theta_0 = 2.):
        self.model = model
        self.d = (self.model.p+1)
        self.range_sigma = np.sqrt(np.var(model.Y)/(self.model.p+1))
        self.nb_quad_term = nb_quad_term
        self.theta_0 = theta_0 
        if 'MRT_distance_sqrt' in self.model.name:
            self.ind_MRT_distance = np.where(np.asarray(self.model.name) == 'MRT_distance_sqrt')[0][0]
        if 'Conv_store_count' in self.model.name:
            self.ind_Conv_store_count = np.where(np.asarray(self.model.name) == 'Conv_store_count')[0][0]
            
    def log_gaussain_prior_intercept(self):
        d= 1
        m_0= np.zeros( 1 )
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
    
        
    def log_gaussain_prior_B(self):
        d= self.d - self.nb_quad_term - 3
        m_0= np.zeros( self.d - self.nb_quad_term -3)
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
    
    def log_gaussain_prior_B_quadratic(self):
        d= self.nb_quad_term
        m_0= np.zeros( self.nb_quad_term )
        # here we would like that the quadratic term are close to zero
        sigma_b_0= self.range_sigma/(2*(2**2))
        
        f = lambda b : -0.5*d * np.log(2 * np.pi) - 0.5*d*np.log( sigma_b_0**2 ) - 0.5 * np.dot((b - m_0).T,(b - m_0)) / (sigma_b_0)**2
        return f
    
    def log_laplace_prior_B(self):
        d= self.d - 1 - 2
        m_0= np.zeros( self.d-1-2)
        b= np.sqrt(self.range_sigma/2)/6
        
        f = lambda x : - d*np.log(2*b) -(1/b)*(np.sum(np.abs(x-m_0)))
        return f
    
    # k_0 =4 for gaussian and 2 if log
    def log_gamma_prior_sigma(self ,k_0=4 ):
        theta_0=self.theta_0
        f = lambda sigma : -k_0*np.log(theta_0)-np.real(scp.special.loggamma(k_0)) + (k_0-1)*np.log(sigma) - sigma/theta_0
        return f
    
    # Force the coefficient to be negative thanks to prior knowledge of real estate
    def log_gamma_prior_expert(self ,k_0=4 ):
        theta_0 = self.theta_0
        f = lambda b : -k_0*np.log(theta_0)-np.real(scp.special.loggamma(k_0)) + (k_0-1)*np.log(-b) + b/theta_0
        return f
    
    def log_joint_prior(self):
        
        expert = [ self.ind_MRT_distance, self.ind_Conv_store_count ]
        other = [ i for i in range(1,self.d-self.nb_quad_term) if i not in expert]
        
        
        log_prior_expert = self.log_gamma_prior_expert()
        log_prior_beta = self.log_gaussain_prior_B()
        log_prior_beta_quad = self.log_gaussain_prior_B_quadratic()
        log_prior_sigma = self.log_gamma_prior_sigma()
        log_prior_intercept = self.log_gaussain_prior_intercept()
        
        if self.nb_quad_term > 0 :
            ans = lambda variable : log_prior_expert(variable) + log_prior_beta(variable[other]) + log_prior_beta_quad(variable[self.d-self.nb_quad_term :self.d]) + log_prior_sigma(variable[-1])
        else :
            ans = lambda variable : log_prior_intercept(variable[0]) + log_prior_expert(variable[self.ind_MRT_distance]) +log_prior_expert(-variable[self.ind_Conv_store_count]) + log_prior_beta(variable[other])  + log_prior_sigma(variable[-1])
            
        return ans
    
    def log_likelihood(self,y,x):
        ans = lambda variable : np.real(
                                    - 0.5* np.log(2 * np.pi) 
                                    - 0.5*np.log( variable[-1]**2 ) 
                                    - 0.5 *((y-np.dot(x,variable[:self.d]))**2) / (variable[-1])**2)
        return ans
    

    
    
    
    
    
    
    
    
    
    
    