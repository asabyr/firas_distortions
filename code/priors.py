import numpy as np

#dT only for CMB-only mocks
def logPrior_dT(theta):
    return 0.0

#dT, y, dust with Bd prior
def logPrior_dust_Bdgauss(theta):
    dT, y, Ad, Bd, Td=theta
    
    if Td>100 or Td<0:
       return -np.inf
    else:
       mu_Bd=1.51
       sigma_Bd=0.1
       gauss_Bd=np.log(1/(np.sqrt(2*np.pi)*sigma_Bd))-0.5*(Bd-mu_Bd)**2/sigma_Bd**2

       return gauss_Bd

#flat priors
def logPrior_dust(theta):
    dT, y, Ad, Bd, Td=theta
    if Td>100 or Td<0 or Bd<0 or Bd>3:
       return -np.inf
    else:
       return 0.0
    
# def logPrior_dust_Bdgauss_SZpack_rel(theta):
#     dT, y, kTe, Ad, Bd, Td=theta
    
#     if Td>100 or Td<0 or kTe<0 or kTe==0 or kTe>70:
#        return -np.inf
#     else:
#        mu_Bd=1.51
#        sigma_Bd=0.1
#        gauss_Bd=np.log(1/(np.sqrt(2*np.pi)*sigma_Bd))-0.5*(Bd-mu_Bd)**2/sigma_Bd**2

#        return gauss_Bd

def logPrior_dust_Bdgauss_SZpack_0to190_rel(theta):
    dT, y, kTe, Ad, Bd, Td=theta

    if Td>100 or Td<0 or kTe<0 or kTe==0 or kTe>190 or y<0 or y==0:
       return -np.inf
    else:
       mu_Bd=1.51
       sigma_Bd=0.1
       gauss_Bd=np.log(1/(np.sqrt(2*np.pi)*sigma_Bd))-0.5*(Bd-mu_Bd)**2/sigma_Bd**2

       return gauss_Bd
