import torch 
import numpy as np

"""
This file defines all the model that we use, written with torch to be able to run GVA and important sampling on it:
   
    A : Gaussian model with gaussina prior 
    B : Gaussina model with laplace prior 
    C : Student model with Gaussian prior
    D : Student model with Laplace prior 
    E : Gaussian model with expert prior 

"""

##############################################################################################################
############################## A : Gaussian model with gaussina prior ########################################
##############################################################################################################

class Gaussian_model_gaussian_prior_torch :
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
        m_0= torch.zeros( d ,1)
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        return f
    
    def log_gaussain_prior_B_quadratic(self):
        d= self.nb_quad_term
        m_0= torch.zeros( d ,1)
        # here we would like that the quadratic term are close to zero
        sigma_b_0= self.range_sigma/(2*(2**2))
        
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        return f
    
    def log_gamma_prior_sigma(self , k_0=torch.tensor(2.), theta_0=torch.tensor(2.)):
        
        f = lambda sigma : -k_0*torch.log(theta_0)-torch.real(torch.lgamma(k_0)) + (k_0-1)*torch.log(sigma) - (sigma)/theta_0
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
        ans = lambda variable : torch.real(
                                    -0.5 * torch.log(torch.tensor(2 * np.pi)) 
                                    -0.5 * torch.log( variable[-1]**2 ) 
                                    -0.5 * ((y-(x@variable[:self.d].double()))**2) / (variable[-1])**2 )
        return ans

##############################################################################################################
############################## B : Gaussina model with laplace prior  ########################################
##############################################################################################################
        
class Gaussian_model_laplace_prior_torch :
    """ 
    the variable need to be like [ slope , slope_quad , sigma ]
    """
    
    def __init__(self,model,nb_quad_term = 0, coeff = 8):
        self.model = model
        self.d = (self.model.p+1)
        self.range_sigma = np.sqrt(np.var(model.Y)/(self.model.p+1))
        self.nb_quad_term = nb_quad_term
        self.coeff = coeff
    
    def log_gaussain_prior_intercept(self):
        d= 1
        m_0= torch.zeros( 1 ,1 )
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        return f
   
    
    def log_laplace_prior_B(self):
        d= self.d-1 - self.nb_quad_term
        m_0= torch.zeros( self.d-1 - self.nb_quad_term ,1)
        b= np.sqrt(self.range_sigma/2)/self.coeff
        
        f = lambda x : - d*torch.log(torch.tensor(2*b)) -(1/b)*(torch.sum((x-m_0).abs(),dim=0))
        return f
    
    def log_laplace_prior_B_quad(self):
        d= self.nb_quad_term
        m_0= torch.zeros( self.nb_quad_term ,1)
        b= np.sqrt(self.range_sigma/2)/(self.coeff*2)
        
        f = lambda x : - d*torch.log(torch.tensor(2*b)) -(1/b)*(torch.sum((x-m_0).abs(),dim=0))
        return f
   
    def log_gamma_prior_sigma(self , k_0=torch.tensor(4.), theta_0=torch.tensor(2.)):
        
        f = lambda sigma : -k_0*torch.log(theta_0)-torch.real(torch.lgamma(k_0)) + (k_0-1)*torch.log(sigma) - (sigma)/theta_0
        return f
    
    def log_likelihood(self,y,x):
        ans = lambda variable : torch.real(
                                    -0.5 * torch.log(torch.tensor(2 * np.pi)) 
                                    -0.5 * torch.log( variable[-1]**2 ) 
                                    -0.5 * ((y-(x@variable[:self.d].double()))**2) / (variable[-1])**2 )
        return ans
    
    def log_joint_prior(self):
        
        log_prior_intercept = self.log_gaussain_prior_intercept()
        log_prior_beta = self.log_laplace_prior_B()
        log_prior_sigma = self.log_gamma_prior_sigma()
        log_prior_beta_quad = self.log_laplace_prior_B_quad()
        
        if self.nb_quad_term > 0 :
            ans = lambda variable : log_prior_intercept(variable[0])+ log_prior_beta(variable[1:self.d-self.nb_quad_term]) + log_prior_beta_quad(variable[self.d-self.nb_quad_term :self.d]) + log_prior_sigma(variable[-1])
        else :
            ans = lambda variable : log_prior_intercept(variable[0])+ log_prior_beta(variable[1:self.d])  + log_prior_sigma(variable[-1])
            
       
        return ans
    
##############################################################################################################
############################## C : Student model with Gaussian prior  ########################################
##############################################################################################################    

class Student_model_gaussian_prior_torch :
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
        m_0= torch.zeros( self.d - self.nb_quad_term,1 )
        sigma_b_0= self.range_sigma
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        
        return f
    
    def log_gaussain_prior_B_quadratic(self):
        d= self.nb_quad_term
        m_0= torch.zeros( self.nb_quad_term,1 )
        # here we would like that the quadratic term are close to zero
        sigma_b_0= self.range_sigma/(2*(2**2))
        
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        return f
    
   
    def log_gamma_prior_nu(self , k_0=torch.tensor(4.), theta_0=torch.tensor(2.)):
        
        f = lambda sigma : -k_0*torch.log(theta_0)-torch.real(torch.lgamma(k_0)) + (k_0-1)*torch.log(sigma) - (sigma)/theta_0
        return f
    
    def log_gamma_prior_sigma(self , k_0=torch.tensor(4.), theta_0=torch.tensor(2.)):
        
        f = lambda sigma : -k_0*torch.log(theta_0)-torch.real(torch.lgamma(k_0)) + (k_0-1)*torch.log(sigma) - (sigma)/theta_0
        return f
    
    def log_likelihood(self,y,x):
        ans = lambda variable : torch.real(
                                    - torch.log(variable[-2])                              # changing sigma changes the density
                                    + torch.lgamma( (variable[-1]+1)/2 )            # normal term of the student density
                                    - 0.5 *torch.log( variable[-1] * np.pi )                  # normal term of the student density
                                    - torch.lgamma( variable[-1]/2 )                # normal term of the student density
                                    - ((variable[-1]+1)/2) * torch.log(1 + ((y-(x@variable[0:self.d].double()))**2) /((variable[-2]**2)*variable[-1]))   )
    
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
############################## D : Student model with Laplace prior  #########################################
############################################################################################################## 

class Student_model_laplace_prior_torch :
    """ 
    the variable need to be set as [ slope , slope_quad , sigma , df ]
    """
    def __init__(self,model,nb_quad_term = 0):
        self.model = model
        self.d = (self.model.p+1)
        self.range_sigma = np.sqrt(np.var(model.Y)/(self.model.p+1))
        self.nb_quad_term = nb_quad_term
    
    def log_gaussain_prior_intercept(self):
        d= 1
        m_0= torch.zeros( 1 )
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        return f
   
    
    def log_laplace_prior_B(self):
        d= self.d-1
        m_0= torch.zeros( self.d-1,1)
        b= np.sqrt(self.range_sigma/2)/6
        
        f = lambda x : - d*torch.log(2*b) -(1/b)*(torch.sum((x-m_0).abs(),dim=0))
        return f
    def log_gamma_prior_nu(self , k_0=torch.tensor(4.), theta_0=torch.tensor(2.)):
        
        f = lambda sigma : -k_0*torch.log(theta_0)-torch.real(torch.lgamma(k_0)) + (k_0-1)*torch.log(sigma) - (sigma)/theta_0
        return f
    
    def log_gamma_prior_sigma(self , k_0=torch.tensor(4.), theta_0=torch.tensor(2.)):
        
        f = lambda sigma : -k_0*torch.log(theta_0)-torch.real(torch.lgamma(k_0)) + (k_0-1)*torch.log(sigma) - (sigma)/theta_0
        return f
    
    def log_likelihood(self,y,x):
        ans = lambda variable : torch.real(
                                    - torch.log(variable[-2])                              # changing sigma changes the density
                                    + torch.lgamma( (variable[-1]+1)/2 )            # normal term of the student density
                                    - 0.5 *torch.log( variable[-1] * np.pi )                  # normal term of the student density
                                    - torch.lgamma( variable[-1]/2 )                # normal term of the student density
                                    - ((variable[-1]+1)/2) * torch.log(1 + ( (y- (x@variable[0:self.d].double()) )**2 ) /((variable[-2]**2)*variable[-1]))   )
    
        return ans
    
    def log_joint_prior(self):
        log_prior_intercept = self.log_gaussain_prior_intercept()
        log_prior_beta = self.log_laplace_prior_B()
        log_prior_sigma = self.log_gamma_prior_sigma()
        log_prior_nu = self.log_gamma_prior_nu()
        
        ans = lambda variable : log_prior_intercept(variable[0])+ log_prior_beta(variable[1:self.d])  + log_prior_sigma(variable[-2]) + log_prior_nu(variable[-1])
            
        return ans


##############################################################################################################
############################## E : Gaussian model with expert prior  #########################################
############################################################################################################## 
   
class Gaussian_model_expert_prior_torch :
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
        m_0= torch.zeros( 1 ,1 )
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        return f
    
        
    def log_gaussain_prior_B(self):
        d= self.d - self.nb_quad_term - 3
        m_0= torch.zeros( self.d - self.nb_quad_term -3,1)
        sigma_b_0= self.range_sigma
        
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        
        return f
    
    def log_gaussain_prior_B_quadratic(self):
        d= self.nb_quad_term
        m_0= torch.zeros( self.nb_quad_term,1 )
        # here we would like that the quadratic term are close to zero
        sigma_b_0= self.range_sigma/(2*(2**2))
        
        f = lambda b : -0.5*d * torch.log(torch.tensor(2 * np.pi)) - 0.5*d*torch.log(torch.tensor(sigma_b_0**2)) - 0.5 * (torch.norm((b - m_0),dim=0)**2) / (sigma_b_0)**2
        
        return f
    
    def log_laplace_prior_B(self):
        d= self.d-1 -3
        m_0= torch.zeros( self.d-1-3 , 1 )
        b= np.sqrt(self.range_sigma/2)/6
        
        f = lambda x : - d*torch.log(torch.tensor(2*b)) -(1/b)*(torch.sum((x-m_0).abs(),dim=0))
        return f
    
    # k_0 =4 for gaussian and 2 if log
    def log_gamma_prior_sigma(self ,k_0= torch.tensor(2.)):
        
        theta_0= torch.tensor(self.theta_0)
        f = lambda sigma : -k_0*torch.log(theta_0)-torch.real(torch.lgamma(k_0)) + 0.5*(k_0-1)*torch.log(sigma**2) - (sigma)/theta_0
        return f
    
    # Force the coefficient to be negative thanks to prior knowledge of real estate
    def log_gamma_prior_expert(self ,k_0= torch.tensor(4.)):
        
        theta_0= torch.tensor(self.theta_0)
        f = lambda b : -k_0*torch.log(theta_0)-torch.real(torch.lgamma(k_0)) + 0.5*(k_0-1)*torch.log(b**2) - (b)/theta_0
        return f
    
    def log_joint_prior(self):
        
        expert = [ self.ind_MRT_distance, self.ind_Conv_store_count]
        other = [ i for i in range(1,self.d-self.nb_quad_term) if i not in expert]

        log_prior_expert = self.log_gamma_prior_expert()
        log_prior_beta = self.log_gaussain_prior_B()
        log_prior_beta_quad = self.log_gaussain_prior_B_quadratic()
        log_prior_sigma = self.log_gamma_prior_sigma()
        log_prior_intercept = self.log_gaussain_prior_intercept()
        
        if self.nb_quad_term > 0 :
            ans = lambda variable : log_prior_expert(variable) + log_prior_beta(variable[other]) + log_prior_beta_quad(variable[self.d-self.nb_quad_term :self.d]) + log_prior_sigma(variable[-1])
        else :
            ans = lambda variable : log_prior_intercept(variable[0]) + log_prior_expert(-variable[self.ind_MRT_distance]) +log_prior_expert(variable[self.ind_Conv_store_count]) + log_prior_beta(variable[other])  + log_prior_sigma(variable[-1])
            
        return ans
    
    def log_likelihood(self,y,x):
        ans = lambda variable : torch.real(
                                    - 0.5* torch.log(torch.tensor(2 * np.pi)) 
                                    - 0.5*torch.log( variable[-1]**2 ) 
                                    - 0.5 *((y-x@variable[:self.d].double())**2) / (variable[-1])**2)
        return ans