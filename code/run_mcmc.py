import numpy as np
import sys
sys.path.append('/moto/home/as6131/firas_distortions/')
sys.path.append('/moto/hill/users/as6131/software/sd_foregrounds/')
import sys

# Set the path to the desired emcee version
sys.path.append('/moto/home/as6131/.local/lib/python3.9/site-packages')

# Import emcee with the specified version
import emcee

from spectral_distortions import *
from foregrounds import *
from minimization_funcs import *
from mcmc_funcs import *
#from mcmc_tensorflow import * 
import configparser
from read_data import *
from findbestfit import FindBestFit
from minimize_priors import *

ini_file_dir='/moto/home/as6131/firas_distortions/mcmc_ini_files/'
data_dir='/moto/home/as6131/firas_distortions/data/'
output_dir='/moto/hill/users/as6131/firas_distortions/mcmc_results/'


ini_file=str(sys.argv[1])
config=configparser.ConfigParser()
config.read(ini_file_dir+ini_file)

###################sampler settings###################

sampler_type=config['general']['sampler']

if sampler_type=='emcee': #tried other samplers too, decided to keep an if statement 
    mcmc_dict={}
    mcmc_dict['n walkers']=config.getint('mcmc','n walkers')
    mcmc_dict['steps']=config.getint('mcmc','steps')
    mcmc_dict['initial param perturb [%]']=config.getfloat('mcmc','initial param perturb [%]') #for setting walker positions
     

###################model parameters###################

param_dict=eval(config.get('model','model components'))
logPrior_str=config.get('model','priors')#from priors.py
logPrior=eval(logPrior_str)
Inu=eval(config.get('model','model function')) #from Inu.py
min_prior=eval(config.get('model','min_bounds')) #from minimization_funcs.py

if config.has_option('model', 'gauss_priors'):
    gausspriors=np.asarray(config.get('model', 'gauss_priors').split(','),dtype=np.float64)
else:
    gausspriors=[]  

###################data parameters###################
data_fname=config['general']['data file name']
mock_data=config['general']['mock or data'] #not used
method=config['general']['method'] #invvar etc.
fsky=config.getint('general','fsky') #20,40,60
nu=config.getfloat('general','frequencies') #0 means lowf only, otherwise the max freq in GHz

#use only Cterm correlation in covariance unless otherwise specified
#not used in the final results
if config.has_option('general', 'covariance file')==True: 
   covariance_file=config['general']['covariance file']
   covariance_type=config['general']['covariance type']
else:
   covariance_file=None
   covariance_type='c'

###################other###################
continue_run=config.getboolean('general','continue run')

#if ini file is in subdirectory, clean the output file name
if '/' in ini_file:
    fname=job_name=output_dir+ini_file.rsplit('/',1)[1].replace(".ini","")
else:
    fname=output_dir+ini_file.replace('.ini','')

if sampler_type=='emcee':
    run_parallel=config.getboolean('general','parallel')

#moves for emcee
if config.getboolean('mcmc','emcee moves')==True:
      mcmc_moves=np.array(config.get('mcmc','emcee moves frac').split(','),dtype=np.float64)
else:
      mcmc_moves=np.array([])

###################prepare data###################
#not useful, but leave for now
if mock_data=='data':
  print("reading data")
elif mock_data=='mock':
  print("reading mock data")

#low freqs
if config.has_option('general', 'remove lines'):
  if config.getboolean('general','remove lines')==False:
    #not used
    low_data_arr=prepare_data_lowf_masked(data_fname, fsky, method, [2,-1])
  else:
    low_data_arr=prepare_data_lowf_masked_nolines(data_fname, fsky, method, [2,-1], 1, covariance_type, covariance_file)   
else:
    #default to removing points near emission lines
    low_data_arr=prepare_data_lowf_masked_nolines(data_fname, fsky, method, [2,-1], 1, covariance_type, covariance_file)

#high freqs
if nu>0.0:  
  if config.has_option('general', 'remove lines'):
    if config.getboolean('general','remove lines')==False:
      #not used
      high_data_arr=prepare_data_highf_masked(data_fname, fsky, method, nu, 3)
    else:
      high_data_arr=prepare_data_highf_masked_nolines(data_fname, fsky, method, nu, 3, 1,covariance_type, covariance_file)
  else:
     #default to removing points near emission lines
     high_data_arr=prepare_data_highf_masked_nolines(data_fname, fsky, method, nu, 3,1,covariance_type, covariance_file)
     
  if config.has_option('general', 'initial_fid')==True:
    #set initial values from Abitbol+2017
    best_fit_initial=params_initial_Abitbol2017(param_dict)
  else:
    #set initial values from minimization 
    bestfit_object=FindBestFit(min_prior, param_dict, low_data_arr, data_high=high_data_arr, print_fit=True, gausspriors=gausspriors)
    best_fit_result=bestfit_object.minimize_lowf_highf()
    best_fit_initial=np.array(best_fit_result['x'])
    print("best fit:")
    print(best_fit_initial)

#low freqs only
else:
  if config.has_option('general', 'initial_fid')==True:
    best_fit_initial=params_initial_Abitbol2017(param_dict)
  else:
    bestfit_object=FindBestFit(min_prior, param_dict, low_data_arr, data_high=None, print_fit=True, gausspriors=gausspriors)
    best_fit_result=bestfit_object.minimize_lowf_highf()
    best_fit_initial=np.array(best_fit_result['x'])
    print("best fit:")
    print(best_fit_initial)

#if best fit values are zero, need to initialize at some small values
inds=np.where(best_fit_initial==0.0)[0]
if len(inds)>0:
  zero_arr=np.zeros(len(best_fit_initial))
  zero_arr[inds]=1e-5
  copy_best_fit=np.copy(best_fit_initial)
  best_fit_initial=copy_best_fit+zero_arr

###################run emcee ###################
if sampler_type=='emcee':
    
    if nu==0:
      print("running low frequencies")
      MCMC_run(logPrior, Inu, param_dict, mcmc_dict, fname, low_data_arr,continue_run, mcmc_moves, run_parallel, False, best_fit_initial)
    elif nu>0.0:
      print("running low and high frequencies")
      MCMC_run(logPrior, Inu, param_dict, mcmc_dict, fname, low_data_arr, continue_run, mcmc_moves, run_parallel, False, best_fit_initial, high_data_arr)
