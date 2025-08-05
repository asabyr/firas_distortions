import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import os
import copy
from scipy import linalg

file_path=os.path.dirname(os.path.abspath(__file__))
if file_path=='/moto/home/as6131/firas_distortions/code':
   fg_path=file_path.replace('/moto/home/as6131/firas_distortions/code','/moto/hill/users/as6131/software/sd_distortions')
else:
   fg_path=file_path.replace('firas_distortions/code','software/sd_distortions')
sys.path.append(fg_path)
dir_path=file_path.replace('/code','')

from spectral_distortions import *
from foregrounds import *
from read_data import *
import constants as const
#from read_nuts_posteriors import read_NUTS_posteriors

#initial amplitudes, rescaled from A17 based on fixed nu0=353GHz
Ad_x=1.36e6
Ad_353=Ad_x*(353.0e9*const.hplanck/(const.kboltz*21.0))**(1.53+3)
Acib_x=3.46e5
Acib_353=Acib_x*(353.0e9*const.hplanck/(const.kboltz*18.8))**(0.86+3)

def log_gauss_prior(sigma, mu, value):
    return np.log(1/(np.sqrt(2*np.pi)*sigma))-0.5*(value-mu)**2/sigma**2

#this function takes care of gaussian priors
#dependent on the models I've tried
#(there is definitely a more general way to code this)
def gauss_2(theta, theta_names, gausspriors):
    """assumes one gaussian prior on either Td or Bd.
    """
    if len(theta)==5:
        assert thermal_dust_rad in theta_names.keys()
        dT, y, Ad, Bd, Td=theta
    elif len(theta)==6:
        if DeltaI_reltSZ in theta_names.keys()\
            or DeltaI_reltSZ_Y3 in theta_names.keys()\
            or DeltaI_reltSZ_Y2 in theta_names.keys()\
            or DeltaI_reltSZ_Y1 in theta_names.keys():
            dT, y, Trel, Ad, Bd, Td=theta
        else:
            #can work for FF, CO and Acib
            dT, y, Ad, Bd, Td, Aextra=theta
    elif len(theta)==7:
        #can be sync or FF+CO
        dT, y, Ad, Bd, Td, As, Bs=theta
    #####for high freqs only #####
    elif len(theta)==8:
        #dust+sync+ CO/FF/fixed CIB/curvature
        dT, y, Ad, Bd, Td, As, Bs, Aextra=theta
    elif len(theta)==9:
        #dust+sync+ plaw or CO+FF
        dT, y, Ad, Bd, Td, As, Bs, Aplaw, Bplaw=theta
    # elif len(theta)==10:
    #     dT, y, Ad, Bd, Td, As, Bs, Acib, Bcib, Tcib=theta

    mu, sigma=gausspriors

    if mu>10:
        # print(f"dust Td prior: {mu}, {sigma}, {Td}")
        logprior=log_gauss_prior(sigma, mu, Td)
        if Ad==0:
            logprior=0.0
    elif mu>0 and mu<10:
        # print(f"dust Bd prior: {mu}, {sigma}, {Bd}")
        logprior=log_gauss_prior(sigma, mu, Bd)
        if Ad==0.0:
            logprior=0.0
    elif mu<0:#only high freqs
        # print(f"sync Bs prior: {mu}, {sigma}, {Bs}")
        logprior=log_gauss_prior(sigma, mu, Bs)
        if As==0.0:
            logprior=0.0
    # print(f"total prior: {logprior}")
    return logprior

def gauss_4(theta, theta_names, gausspriors):
    """assumes 2 gaussian priors: either (Bd and Bs) or (Bcib and Tcib)
    """
    mu1, sigma1, mu2, sigma2=gausspriors
    #both cib priors
    # if cib_rad in theta_names.keys():
    #     if len(theta)==10:
    #         dT, y, Ad, Bd, Td, As, Bs, Acib, Bcib, Tcib=theta
    #     elif len(theta)==8:
    #         dT, y, Ad, Bd, Td, Acib, Bcib, Tcib=theta

    #     if Acib!=0:
    #         logprior1=log_gauss_prior(sigma1, mu1, Bcib)
    #         logprior2=log_gauss_prior(sigma1, mu1, Tcib)
    #         logprior=logprior1+logprior2
    #     else:
    #         logprior=0.0
    #dust + sync priors
    if cib_rad not in theta_names.keys():
        if len(theta)==8:
            dT, y, Ad, Bd, Td, Aextra, As, Bs=theta
        elif len(theta)==7:#low freqs
            # print("fitting dust + sync")
            dT, y, Ad, Bd, Td, As, Bs=theta
        else:
            sys.exit('check input model, needs to include sync and dust')
        
        if Ad==0.0:
            logprior1=0.0
        else:
            if mu1>10:
                logprior1=log_gauss_prior(sigma1, mu1, Td)
                # print(f"dust Td prior, {logprior1}: {mu1}, {sigma1}, {Td}")
            else:
                logprior1=log_gauss_prior(sigma1, mu1, Bd)
                # print(f"dust Bd prior, {logprior1}: {mu1}, {sigma1}, {Bd}")
        if As==0.0:
            logprior2=0.0
        else:
            logprior2=log_gauss_prior(sigma2, mu2, Bs)
            # print(f"sync Bs prior, {logprior2}: {mu2}, {sigma2}, {Bs}")

        logprior=logprior1+logprior2
    # print(f"total prior: {logprior}")
    return logprior

#-lnP function for low frequencies
def log_post_func_variable(theta, theta_names, cov_inv, data_nu, data_Inu, gausspriors=[]):

    #logL
    model=total_Inu_variable(data_nu, theta_names, theta)
    delta_x=data_Inu-model
    step1=np.dot(delta_x.T,cov_inv)
    step2=np.dot(step1,delta_x)

    if not np.isfinite(step2):
        return -np.inf

    if len(gausspriors)==2:
        logprior=gauss_2(theta, theta_names, gausspriors)
    elif len(gausspriors)==4:
        logprior=gauss_4(theta, theta_names, gausspriors)
    # elif len(gausspriors)==6:
    #     logprior=gauss_6(theta, theta_names, gausspriors)
    elif len(gausspriors)==0:
        logprior=0.0
    else:
        sys.exit("incorrect gaussian prior array")

    #return negative logPosterior
    if logprior!=0:
        return -(-0.5*(step2)+logprior)
    else:
        return 0.5*step2

#chi^2 function for low frequencies
def chi2_sq_func_variable(theta, theta_names, cov_inv, data_nu, data_Inu):

    #logL
    model=total_Inu_variable(data_nu, theta_names, theta)
    delta_x=data_Inu-model
    step1=np.dot(delta_x.T,cov_inv)
    step2=np.dot(step1,delta_x)

    return step2

#same funcs for low + high frequencies 
def log_post_func_joint_variable(theta, theta_names, cov_inv_1, data_nu_1, data_Inu_1,
                                cov_inv_2,data_nu_2, data_Inu_2, gausspriors=[]):

    model_1=total_Inu_variable(data_nu_1, theta_names, theta)
    delta_x_1=data_Inu_1-model_1
    step1_1=np.dot(delta_x_1.T,cov_inv_1)
    step2_1=np.dot(step1_1,delta_x_1)

    model_2=total_Inu_variable(data_nu_2,theta_names, theta)
    delta_x_2=data_Inu_2-model_2
    step1_2=np.dot(delta_x_2.T,cov_inv_2)
    step2_2=np.dot(step1_2,delta_x_2)

    if not np.isfinite(step2_1):
        return -np.inf
    if not np.isfinite(step2_2):
        return -np.inf

    if len(gausspriors)==2:
        logprior=gauss_2(theta, theta_names, gausspriors)
    elif len(gausspriors)==4:
        logprior=gauss_4(theta, theta_names, gausspriors)
    # elif len(gausspriors)==6:
    #     logprior=gauss_6(theta, theta_names, gausspriors)
    elif len(gausspriors)==0:
        logprior=0.0
    else:
        sys.exit("incorrect gaussian prior array")

    if logprior!=0:
        return -(-0.5*(step2_1+step2_2)+logprior)
    else:
        return 0.5*(step2_1+step2_2)

def chi2_sq_func_joint_variable(theta, theta_names, cov_inv_1, data_nu_1, data_Inu_1,
                                cov_inv_2,data_nu_2, data_Inu_2):

    model_1=total_Inu_variable(data_nu_1, theta_names, theta)
    delta_x_1=data_Inu_1-model_1
    step1_1=np.dot(delta_x_1.T,cov_inv_1)
    step2_1=np.dot(step1_1,delta_x_1)

    model_2=total_Inu_variable(data_nu_2,theta_names, theta)
    delta_x_2=data_Inu_2-model_2
    step1_2=np.dot(delta_x_2.T,cov_inv_2)
    step2_2=np.dot(step1_2,delta_x_2)

    return step2_1+step2_2

#model function (made up of any functions, just by adding up)
def total_Inu_variable(nu, params_names, params):

    counter=0
    component_count=0

    for param in params_names:

        if params_names[param]==1:

            one_component=param(nu, params[counter])
            counter+=1

        elif params_names[param]==2:

            one_component=param(nu, params[counter],params[counter+1])
            counter+=2

        elif params_names[param]==3:

            one_component=param(nu, params[counter],params[counter+1],params[counter+2])
            counter+=3

        elif params_names[param]==4:

            one_component=param(nu, params[counter],params[counter+1],params[counter+2],params[counter+3])
            counter+=4

        component_count+=1

        if component_count==1:
            totmodel=one_component
        else:
            totmodel+=one_component

    return totmodel

#fiducial parameters from A17
def params_initial_Abitbol2017(params_names):

    x0=[]
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append(1.2e-4*const.TCMB)
        elif param==DeltaI_y:
            x0.append(1.77e-6)
        elif param==DeltaI_reltSZ:
            x0.append(1.77e-6)
            x0.append(1.245)
        elif param==DeltaI_rel_SZpack:
            x0.append(1.77e-6)
            x0.append(1.245)
	#dust
        elif param==thermal_dust_rad:
            x0.append(Ad_353)
            x0.append(1.53)
            x0.append(21.)
        #moments
        elif param==dust_moments_omega2_omega3:
            x0.append(Ad_353)
            x0.append(0.1)
            x0.append(0.1)
        elif param==dust_moments_omega2_omega3_bestfit:
            x0.append(Ad_353)
            x0.append(0.1)
            x0.append(0.1)
        elif param==dust_moments_omega2_omega3_omega22:
            x0.append(Ad_353)
            x0.append(0.1)
            x0.append(0.1)
            x0.append(0.1)
        #cib
        # elif param==cib_rad:
        #     x0.append(Acib_353)
        #     x0.append(0.86)
        #     x0.append(18.8)
        elif param==cib_rad_MH23:
            x0.append(Acib_353)
        # elif param==cib_rad_A17:
        #     x0.append(Acib_x)
        #synch
        elif param==jens_synch_rad:
            x0.append(288.)
            x0.append(-0.82)
            x0.append(0.2)
        elif param==jens_synch_rad_no_curv:
            x0.append(288.)
            x0.append(-0.82)
        #FF
        elif param==jens_freefree_rad:
            x0.append(300.)
        #some power law
        elif param==powerlaw:
            x0.append(300.)
            x0.append(1.0)
        elif param==co_rad:
            x0.append(1.0)
    print("intial params")
    print(x0)
    return np.array(x0)

#initialize from a dictionary
def params_initial(params_names):

    x0=[]
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append(params_names[DeltaI_DeltaT])
        elif param==DeltaI_y:
            x0.append(params_names[DeltaI_y])
        # elif param==DeltaI_reltSZ:
        #     x0.append(params_names[DeltaI_reltSZ][0])
        #     x0.append(params_names[DeltaI_reltSZ][1])
	    #dust
        elif param==thermal_dust_rad:
            x0.append(params_names[thermal_dust_rad][0])
            x0.append(params_names[thermal_dust_rad][1])
            x0.append(params_names[thermal_dust_rad][2])
        #synch
        # elif param==jens_synch_rad_no_curv:
        #     x0.append(params_names[jens_synch_rad_no_curv][0])
        #     x0.append(params_names[jens_synch_rad_no_curv][1])
        # elif param==jens_freefree_rad:
        #     x0.append(params_names[jens_freefree_rad])
    print("intial params")
    print(x0)
    return np.array(x0)

#read best fit parameters from minimization runs
def read_bestfit(params_names, best_fit):

    i=0
    for param in params_names:

        if param==DeltaI_DeltaT:
            print(f"dT:{best_fit[i]}")
            i+=1
        elif param==DeltaI_y:
            print(f"y:{best_fit[i]}")
            i+=1
        elif param==DeltaI_reltSZ:
            print(f"y:{best_fit[i]}")
            print(f"Trel:{best_fit[i+1]}")
            i+=2
        elif param==thermal_dust_rad:
            print(f"dust amp:{best_fit[i]}")
            print(f"dust index:{best_fit[i+1]}")
            print(f"dust temp:{best_fit[i+2]}")
            i+=3
        # elif param==cib_rad:
        #     print(f"cib amp:{best_fit[i]}")
        #     print(f"cib index:{best_fit[i+1]}")
        #     print(f"cib temp:{best_fit[i+2]}")
        #     i+=3
        elif param==jens_synch_rad_no_curv:
            print(f"sync amp:{best_fit[i]}")
            print(f"sync index:{best_fit[i+1]}")
            i+=2
        elif param==jens_synch_rad:
            print(f"sync amp:{best_fit[i]}")
            print(f"sync index:{best_fit[i+1]}")
            print(f"sync curv:{best_fit[i+2]}")
            i+=3
        elif param==powerlaw:
            print(f"power law amp:{best_fit[i]}")
            print(f"power law index:{best_fit[i+1]}")
            i+=2
        elif param==co_rad:
            print(f"CO amp:{best_fit[i]}")
            i+=1
        elif param==jens_freefree_rad:
            print(f"FF amp:{best_fit[i]}")
            i+=1
        # elif param==cib_rad_A17:
        #     print(f"CIB amp:{best_fit[i]}")
        #     i+=1
        elif param==cib_rad_MH23:
            print(f"CIB amp:{best_fit[i]}")
            i+=1
        elif param==dust_moments_omega2_omega3_omega22:
            print(f"dust amp:{best_fit[i]}")
            print(f"omega2:{best_fit[i+1]}")
            print(f"omega3:{best_fit[i+2]}")
            print(f"omega22:{best_fit[i+3]}")
            i+=4
        elif param==dust_moments_omega2_omega3:
            print(f"dust amp:{best_fit[i]}")
            print(f"omega2:{best_fit[i+1]}")
            print(f"omega3:{best_fit[i+2]}")
            i+=3
        elif param==dust_moments_omega2_omega3_bestfit:
            print(f"dust amp:{best_fit[i]}")
            print(f"omega2:{best_fit[i+1]}")
            print(f"omega3:{best_fit[i+2]}")
            i+=3

#perturb initial parameters between 0.8 and 1.2 of the value randomly
def param_random(params, seed_n, max_min=[0.8,1.2]):

    np.random.seed(seed=seed_n)
    rand_params=np.random.uniform(low=max_min[0]*params, high=max_min[-1]*params)

    return rand_params

def initialize_min(custom_x0, param_dict):

    #default to A17 initial values
    if len(custom_x0)==0:
        initial_x0_orig=params_initial_Abitbol2017(param_dict)

    #convert from dictionary to an array if needed
    elif len(custom_x0)>0:
        if isinstance(custom_x0, np.ndarray)==False:
            initial_x0_orig=params_initial(custom_x0)
        else:
            initial_x0_orig=custom_x0.copy()

    return initial_x0_orig

# #edit table into latex style
# def txt_to_latex(fname):

#     with open(fname, 'r') as txt_file:
#         with open(fname.replace('.txt','_latex.txt'), 'w') as latex_file:

#             for line in txt_file:
#                 if line.startswith("#"):
#                     latex_file.write(line)
#                 else:
#                     newline=line.strip().replace(' ','&')
#                     newline=newline+"\\n"+"\n"
#                     latex_file.write(newline)
#                     print(newline)

#testing for systematic shift in high frequency data
def log_post_func_joint_amp(theta, cov_inv_1, data_nu_1, data_Inu_1,
                          cov_inv_2,data_nu_2, data_Inu_2, gausspriors=[]):

    model_1=total_Inu_low(data_nu_1,  theta)
    delta_x_1=data_Inu_1-model_1
    step1_1=np.dot(delta_x_1.T,cov_inv_1)
    step2_1=np.dot(step1_1,delta_x_1)

    model_2=total_Inu_high(data_nu_2,theta)
    delta_x_2=data_Inu_2-model_2
    step1_2=np.dot(delta_x_2.T,cov_inv_2)
    step2_2=np.dot(step1_2,delta_x_2)

    if len(gausspriors)==2:
        #hardcode dust model gauss priors

        if len(theta)==6:
            dT, y, Ad, Bd, Td, Aextra=theta
        # elif len(theta)==5:
        #     dT, y, Ad, Bd, Td=theta

        mu, sigma=gausspriors
        if Ad==0:
            logprior=0.0
        else:
            if mu>10:
                logprior=log_gauss_prior(sigma, mu, Td)
            else:
                logprior=log_gauss_prior(sigma, mu, Bd)

        return -(-0.5*(step2_1+step2_2)+logprior)
    
    else:
        return 0.5*(step2_1+step2_2)

def chi2_sq_func_joint_amp(theta, cov_inv_1, data_nu_1, data_Inu_1,
                          cov_inv_2,data_nu_2, data_Inu_2):

    model_1=total_Inu_low(data_nu_1,  theta)
    delta_x_1=data_Inu_1-model_1
    step1_1=np.dot(delta_x_1.T,cov_inv_1)
    step2_1=np.dot(step1_1,delta_x_1)

    model_2=total_Inu_high(data_nu_2,theta)
    delta_x_2=data_Inu_2-model_2
    step1_2=np.dot(delta_x_2.T,cov_inv_2)
    step2_2=np.dot(step1_2,delta_x_2)

    return step2_1+step2_2


def total_Inu_low(nu, params):
    dT, y, Ad, Bd, Td, Aextra=params

    blackbody=DeltaI_DeltaT(nu,dT)
    y_dist=DeltaI_y(nu,y)
    dust=thermal_dust_rad(nu, Ad, Bd, Td)

    return blackbody+y_dist+dust

def total_Inu_high(nu, params):
    dT, y, Ad, Bd, Td, Aextra=params

    blackbody=DeltaI_DeltaT(nu,dT)
    y_dist=DeltaI_y(nu,y)
    dust=thermal_dust_rad(nu, Ad, Bd, Td)

    return Aextra*(blackbody+y_dist+dust)
