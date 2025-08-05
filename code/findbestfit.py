import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
import os
import sys
file_path=os.path.dirname(os.path.abspath(__file__))
dir_path=file_path.replace('/code','')
print(dir_path)
sys.path.append(dir_path)
from minimization_funcs import *
# from minimization_funcs import chi_sq_func_variable, chi_sq_func_joint_variable
from minimization_funcs import log_post_func_variable, log_post_func_joint_variable, log_post_func_joint_amp


class FindBestFit:
    
    def __init__(self, params_priors, param_dict, data_low, data_high=None,
                  round_min=0, min_method='Nelder-Mead', min_maxiter=100000, min_tol=1e-6,
                  gausspriors=[],custom_x0={}, initial_points=10, perturb_range=[0.5,1.5],
                  plot=True, print_fit=True, sigma=0., plot_lines=False, Ahigh=False):
        
        #default to just low frequncies
        self.params_priors=params_priors
        self.param_dict=param_dict
        self.data_low=data_low
        self.data_high=data_high
        
        #related to minimization
        self.round_min=round_min # set to non-zero int if you want to round to "round_min" digits when comparing consecutive chi2
        self.min_method=min_method
        self.min_maxiter=min_maxiter
        self.min_tol=min_tol

        #priors/initial values
        self.gausspriors=gausspriors
        self.custom_x0=custom_x0 
        self.initial_points=initial_points #int, from how many points to initialize 
        self.perturb_range=perturb_range #

        #plotting/printing
        self.plot=plot #bool, to plot best-fit results or not
        self.print_fit=print_fit #bool, print the final parameter best-fit values
        self.sigma=sigma #int, if >0 then will print which points have residuals between model & data that are >"sigma"\sigma.
        self.plot_lines=plot_lines #bool, to plot emission lines or not
        
        self.Ahigh=Ahigh

    def minimize_lowf_highf(self):
        
        self.set_up() #set priors and x0
        self.minimize_multiple() #minimize from various points and iteratively
        self.calc_model_residuals() #calculate residuals
        
        if self.plot==True:
            self.plot_bestfit()
        if self.sigma>0:
            self.print_sigma()
        return self.best_fit, self.best_chi2_true  #return best fit dictionary & chi2 (note, not -logPosterior)
    
    def initialize_min(self):

        #default to A17 initial values
        if len(self.custom_x0)==0:
            initial_x0_orig=params_initial_Abitbol2017(self.param_dict)

        #convert from dictionary to an array if needed
        elif len(self.custom_x0)>0:
            if isinstance(self.custom_x0, np.ndarray)==False:
                initial_x0_orig=params_initial(self.custom_x0)
            else:
                initial_x0_orig=copy.deepcopy(self.custom_x0)

        return initial_x0_orig

    def set_up(self):
        if self.Ahigh==True:
            self.priors=self.params_priors #need to specify in the correct array format
        else:
            self.priors=self.params_priors(self.param_dict)
        self.initial_x0_orig=self.initialize_min()

    def minimize_multiple(self):
        
        min_chi2s=np.zeros(self.initial_points)
        bestfits={}

        for point in range(self.initial_points):

            if point==0:
                initial_x0=copy.deepcopy(self.initial_x0_orig)
            else:
                initial_x0=param_random(self.initial_x0_orig, seed_n=point, max_min=self.perturb_range)

            chi2_min, chi2_min_value=self.iterative_min(initial_x0)
            
            if np.isinf(chi2_min_value)==True:
                min_chi2s[point]=1e6
                print(f"best fit chi2 from point {point}")
                print(chi2_min_value)
                bestfits[point]=copy.deepcopy(chi2_min)
            else:
                min_chi2s[point]=copy.deepcopy(chi2_min_value)
                print(f"best fit chi2 from point {point}")
                print(chi2_min_value)
                bestfits[point]=copy.deepcopy(chi2_min)
        
        # if np.isinf(min_chi2s).any()==True:
        #     min_chi2s=np.nan_to_num(min_chi2s, neginf=1e6)
        #     print("negative infinity")
        best_chi2=np.amin(min_chi2s)
        print('best chi2 of all:')
        print(best_chi2)
        best_chi2_ind=np.argmin(min_chi2s)
        self.best_fit=copy.deepcopy(bestfits[best_chi2_ind])
        
        if self.data_high==None:
            self.best_chi2_true=chi2_sq_func_variable(self.best_fit['x'], self.param_dict, self.data_low['cov_inv'],self.data_low['freqs'],self.data_low['intensity'])
        else:
            if self.Ahigh==True:
                self.best_chi2_true=chi2_sq_func_joint_amp(self.best_fit['x'], self.data_low['cov_inv'],self.data_low['freqs'],self.data_low['intensity'],
                                                                self.data_high['cov_inv'],self.data_high['freqs'],self.data_high['intensity'])
            else:
                self.best_chi2_true=chi2_sq_func_joint_variable(self.best_fit['x'], self.param_dict, self.data_low['cov_inv'],self.data_low['freqs'],self.data_low['intensity'],
                                                                self.data_high['cov_inv'],self.data_high['freqs'],self.data_high['intensity'])

        if self.print_fit==True:
            print("best fit:")
            read_bestfit(self.param_dict, self.best_fit['x'])
        
        # return self.best_fit

        
    def iterative_min(self, initial_x0):

        #figure out initial point
        chi2_min_compare,chi2_min_compare_value=self.minimize_low_or_both(initial_x0)

        compare=0
        count_same=0.

        while compare!=1:

            x0=np.array(chi2_min_compare['x'])
            chi2_min,chi2_min_value=self.minimize_low_or_both(x0) #minimize from best-fit
            
            #below: count consecutive equal chi2 to make sure we've reached the minima point
            if self.round_min>0:#whether to round chi2 when comparing or not
                if np.round(chi2_min['fun'],self.round_min)==np.round(chi2_min_compare_value, self.round_min):
                    count_same+=1
            else:
                if chi2_min['fun']==chi2_min_compare_value:
                    count_same+=1

            if count_same==10:#exit loop if 10 chi2 values are the same
                compare=1
            
            #re-set which chi2 value to compare to
            chi2_min_compare=copy.deepcopy(chi2_min)
            chi2_min_compare_value=copy.deepcopy(chi2_min['fun'])
            #print the chi2
            print(chi2_min_compare_value)
        return chi2_min_compare, chi2_min_compare_value
    
    def minimize_low_or_both(self, x0):
        
        if self.data_high==None:
            #low data only
            chi2_min=minimize(log_post_func_variable, x0, args=(self.param_dict, self.data_low['cov_inv'],self.data_low['freqs'],self.data_low['intensity'], self.gausspriors),
                                method=self.min_method,options={'maxiter': self.min_maxiter},tol=self.min_tol, bounds=self.priors)
        else:
            if self.Ahigh==True:
                chi2_min=minimize(log_post_func_joint_amp, x0, args=(self.data_low['cov_inv'],self.data_low['freqs'],self.data_low['intensity'],
                            self.data_high['cov_inv'],self.data_high['freqs'],self.data_high['intensity'], self.gausspriors),
                                method=self.min_method,options={'maxiter': self.min_maxiter},tol=self.min_tol, bounds=self.priors)
            else:
                chi2_min=minimize(log_post_func_joint_variable, x0, args=(self.param_dict, self.data_low['cov_inv'],self.data_low['freqs'],self.data_low['intensity'],
                            self.data_high['cov_inv'],self.data_high['freqs'],self.data_high['intensity'], self.gausspriors),
                                method=self.min_method,options={'maxiter': self.min_maxiter},tol=self.min_tol, bounds=self.priors)
        #return full output and chi2
        return chi2_min, chi2_min['fun']
    
    def calc_model_residuals(self):

        if self.Ahigh==True:
            #calculate sky model & residuals
            self.model_low=total_Inu_low(self.data_low['freqs'],self.best_fit['x'])
            self.sigma_residuals_low=(self.data_low['intensity']-self.model_low)/self.data_low['error']

            if self.data_high!=None:
                self.model_high=total_Inu_high(self.data_high['freqs'],self.best_fit['x'])
                self.sigma_residuals_high=(self.data_high['intensity']-self.model_high)/self.data_high['error']

        else:
            #calculate sky model & residuals
            self.model_low=total_Inu_variable(self.data_low['freqs'],self.param_dict,self.best_fit['x'])
            self.sigma_residuals_low=(self.data_low['intensity']-self.model_low)/self.data_low['error']

            if self.data_high!=None:
                self.model_high=total_Inu_variable(self.data_high['freqs'],self.param_dict,self.best_fit['x'])
                self.sigma_residuals_high=(self.data_high['intensity']-self.model_high)/self.data_high['error']

    def plot_bestfit(self):
        #this is a plot just to visualize the fit
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]},figsize=(10,10))

        ax1.errorbar(self.data_low['freqs']/(1e9),self.data_low['intensity'], yerr=self.data_low['error'], marker='.', ls='')
        ax1.plot(self.data_low['freqs']/(1e9),self.model_low)
        ax2.scatter(self.data_low['freqs']/(1.e9), (self.data_low['intensity']-self.model_low)/self.data_low['error'])

        if self.data_high!=None:
            ax1.errorbar(self.data_high['freqs']/(1e9),self.data_high['intensity'], yerr=self.data_high['error'], marker='.', ls='')
            ax1.plot(self.data_high['freqs']/(1e9),self.model_high)
            ax2.scatter(self.data_high['freqs']/(1.e9), (self.data_high['intensity']-self.model_high)/self.data_high['error'])

        ax2.set_ylim([-7.5,7.5])
        ax2.axhline(-2.5, 0, 1, ls='--', color='black')
        ax2.axhline(2.5, 0, 1, ls='--', color='black')

        if self.data_high!=None:
            ax2.set_xlim([60,1900])
            ax1.set_xlim([60,1900])
        else:
            ax2.set_xlim([60,650])
            ax1.set_xlim([60,650])

        plt.xlabel('freq [GHz]')
        plt.xlabel('intensity [Jy/sr]')

        ax1.set_ylabel(r'$\Delta I_{\nu}$ [MJy/sr]',labelpad=20)
        ax2.set_ylabel(r'residuals/$\sigma$',labelpad=5)
        ax2.set_xlabel(r'$\nu$ [GHz]')
        ax1.set_xscale('log')
        ax2.set_xscale('log')

        if self.plot_lines==True:
            for line in const.spectral_lines:
                ax2.axvline(line,0, 1)

    def print_sigma(self):
        print(f"sigmas > {self.sigma}")
        print(self.sigma_residuals_low[np.where(self.sigma_residuals_low>self.sigma)[0]])
        print(f"frequencies with sigma > {self.sigma}")
        print(self.data_low['freqs'][np.where(self.sigma_residuals_low>self.sigma)[0]])

        if self.data_high!=None:
            print(f"sigmas > {self.sigma}")
            print(self.sigma_residuals_high[np.where(self.sigma_residuals_high>self.sigma)[0]])
            print(f"frequencies with sigma > {self.sigma}")
            print(self.data_high['freqs'][np.where(self.sigma_residuals_high>self.sigma)[0]])