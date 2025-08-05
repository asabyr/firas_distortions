import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy
from scipy import stats

file_path=os.path.dirname(os.path.abspath(__file__))
dir_path=file_path.replace('/code','')
sys.path.append(dir_path)

from read_data import *
import constants as const
from read_nuts_posteriors import read_NUTS_posteriors
from findbestfit import FindBestFit

#count degrees of freedom
def totdeg_both(n_params, data1, data2):
    
    freq1=len(np.array(data1['freqs']))
    freq2=len(np.array(data2['freqs']))
    
    return freq1+freq2-n_params

def totdeg_one(n_params, data1):
    
    freq1=len(np.array(data1['freqs']))

    return freq1-n_params

#count free parameters
def count_parameters(param_dict):

    tot_params=0
    for key in param_dict.keys():
        tot_params+=param_dict[key]
        
    return tot_params

#make initial parameter array for minimization (read from posteriors or previous best-fit)
def x0_for_chi2(custom_x0, param_dict, i, num_param, priors):
    
    if len(custom_x0)>0:
        #print(custom_x0)
        if custom_x0[0].dtype==np.float64:
            
            custom_x0_one=custom_x0[i]
#             print(custom_x0_one)
            if len(custom_x0_one)<num_param: #fill in zeros for additional parameters
#                 print("appending")
                extra=np.zeros(num_param-len(custom_x0_one))
                custom_x0_one=np.append(custom_x0_one, extra)
#                 print(custom_x0_one)
        else:
            custom_x0_one_file=custom_x0[i]
            custom_x0_one_object=read_NUTS_posteriors(fname=custom_x0_one_file, param_dict=param_dict, priors=priors)
            custom_x0_one=custom_x0_one_object.read_mean()
            print("initial x0, based on posterior")
            print(custom_x0_one)
    else:
        custom_x0_one=[]
#     print(custom_x0_one)
    return custom_x0_one

#helpful function to loop through sky fractions 
def calc_chi2_data(prior_func,
                   chi2filename,
                   data_file,
                   param_dict,
                   print_chi2=True,
                  round_min=False, 
                  min_method='Nelder-Mead', priors=[], mask_low=[2,-1], mask_high=3,
                  custom_x0=[], high_freqs=0, sky_fracs=np.array([20, 40, 60]), Ahigh=False, 
                  method='invvar', return_chi2=False, delta_chi2=np.array([0,0,0]),
                  lines_remove=True, priors_mcmc=[]):
    
    
    # method='invvar'
    best_fits=[]
    chi2_sky=[]
    log_post_sky=[]
    i=0 
    
    for fsky in sky_fracs:
        
        #number of parameters and initial x0
        num_param=count_parameters(param_dict)
        custom_x0_one=x0_for_chi2(custom_x0=custom_x0, param_dict=param_dict,
                                  i=i, num_param=num_param, priors=priors_mcmc)
        print(custom_x0_one)
#         break
        
        if lines_remove==True:
            #read lowf data
            healpix_ost_T_lowf=prepare_data_lowf_masked_nolines(fname=data_file, sky_frac=fsky, 
                                                                method=method, ind_mask=mask_low, thresh=1)
        elif lines_remove==False:
            healpix_ost_T_lowf=prepare_data_lowf_masked(fname=data_file, sky_frac=fsky, 
                                                                method=method, ind_mask=mask_low)
        #minimize just low
        if high_freqs==0:
                        
            bestfit_object=FindBestFit(prior_func, param_dict, healpix_ost_T_lowf, data_high=None, round_min=round_min, min_method=min_method, 
                                  min_tol=1e-6, plot=True, print_fit=True, plot_lines=True, gausspriors=priors,custom_x0=custom_x0_one, Ahigh=Ahigh)
            bestfit, best_fit_chi2=bestfit_object.minimize_lowf_highf()
            dof=totdeg_one(num_param, healpix_ost_T_lowf)
        #minimize both low and high
        
        else:
            if lines_remove==True:
                healpix_ost_T_highf=prepare_data_highf_masked_nolines(fname=data_file, sky_frac=fsky, 
                                                                method=method, cutoff_freq=high_freqs, ind_mask=mask_high, thresh=1)
            elif lines_remove==False:
                healpix_ost_T_highf=prepare_data_highf_masked(fname=data_file, sky_frac=fsky, 
                                                                method=method, cutoff_freq=high_freqs, ind_mask=mask_high)
                
            bestfit_object=FindBestFit(prior_func, param_dict, healpix_ost_T_lowf, data_high=healpix_ost_T_highf, round_min=round_min, min_method=min_method, 
                                  min_tol=1e-6, plot=True, print_fit=True, plot_lines=True, gausspriors=priors,custom_x0=custom_x0_one,Ahigh=Ahigh)
            bestfit, best_fit_chi2=bestfit_object.minimize_lowf_highf()
            if Ahigh==True:
                dof=totdeg_both(num_param, healpix_ost_T_lowf,healpix_ost_T_highf)
                dof=dof-1
            else:
                dof=totdeg_both(num_param, healpix_ost_T_lowf,healpix_ost_T_highf)

        
        #collect chi2 and bestfit dictionaries
        print(f'fksy:{fsky}')
        print(f'DOF:{dof}')
        # chi2_sky.append(bestfit['fun'])
        log_post_sky.append(bestfit['fun'])
        chi2_sky.append(best_fit_chi2)
        best_fits.append(np.array(bestfit['x']))
        i+=1

    #save chi2
    chi2_sky_arr_orig=np.array(chi2_sky)
    chi2_sky_arr_orig[~np.isfinite(chi2_sky_arr_orig)] = 0
    # np.savetxt(chi2filename, np.column_stack(chi2_sky_arr), fmt='%.1f')
    pte_sky_arr=1-stats.chi2.cdf(chi2_sky_arr_orig, dof)
    chi2_sky_arr=chi2_sky_arr_orig-delta_chi2
    #print(chi2_sky_arr)
    #print(dof)
    print("final PTE:")
    print(pte_sky_arr)
    np.savetxt(chi2filename, np.column_stack(log_post_sky),fmt='%.1f')
    with open(chi2filename, "a") as f:
        np.savetxt(f, np.column_stack(chi2_sky_arr), fmt='%.1f')
        np.savetxt(f, np.column_stack(pte_sky_arr),fmt='%.4f')
    
    if return_chi2==True:
        return np.array(best_fits), chi2_sky_arr
    #return best fits
    return np.array(best_fits)
