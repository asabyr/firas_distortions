import sys
sys.path.append('/moto/home/as6131/firas_distortions/')
sys.path.append('/moto/hill/users/as6131/software/sd_foregrounds/')
from read_data import *
from numpyro_funcs import *
from numpyro_models import *
# from minimization_funcs import *
import numpy as np
import numpyro 
import jax.numpy as jnp
import configparser
import pickle 
from findbestfit import FindBestFit
from minimize_priors import *

ini_file_dir='/moto/home/as6131/firas_distortions/mcmc_ini_files_NUTS/'
data_dir='/moto/home/as6131/firas_distortions/data/'
output_dir='/moto/hill/users/as6131/firas_distortions/mcmc_results_NUTS/'

#read .ini file
ini_file=str(sys.argv[1])
config=configparser.ConfigParser()
config.read(ini_file_dir+ini_file)

###################data parameters###################
data_fname=config['general']['data file name']
mock_data=config['general']['mock or data'] # not used
method=config['general']['method'] # invvar, invcov_mod
fsky=config.getint('general','fsky') #20,40,60
nu=config.getfloat('general','frequencies') #0 means lowf only, otherwise the max freq in GHz

gpu=config.getboolean('general','gpu') #only cpu version currently works
if gpu==False:
   numpyro.set_platform("cpu")

###################model parameters###################
model=eval(config.get('model', 'model function'))
min_priors=eval(config.get('model','min_priors'))
max_priors=eval(config.get('model','max_priors'))

###################sampler settings###################
num_warmup=config.getint('mcmc', 'num_warmup')
num_samples=config.getint('mcmc', 'num_samples')
max_tree_depth=config.getint('mcmc', 'max_tree_depth')
num_chains=config.getint('mcmc', 'num_chains')
init_type=config.get('mcmc', 'init_type')
target_accept_prob=config.getfloat('mcmc','target_accept_prob')
dense_mass=config.getboolean('mcmc','dense_mass')

#'feasible' is usually used, in which case, we don't need to specify initial parameters
if init_type=="custom":
    init_dict=eval(config.get('model','init_dict'))
    init_dict_value=rescale_to_unit(init_dict, min_priors, max_priors)
else:
    init_dict_value={}

fname=output_dir+ini_file.replace('.ini','.pkl')

#this was used previously, not it's not really needed for anything
if mock_data=='data':
    print("reading data")
elif mock_data=='mock':
    print("reading mock")
else:
    print("need to pick data or mock")
    sys.exit(0)


low_data_arr=prepare_data_lowf_masked_nolines(data_fname, fsky, method, [2,-1], 1)

if nu!=0:
    high_data_arr=prepare_data_highf_masked_nolines(data_fname, fsky, method, nu, 3,1)
    
    #only used in testing
    if init_type=="best_fit":
        param_dict=eval(config.get('model','model components'))
        min_prior=eval(config.get('model','min_bounds'))
        if config.has_option('model', 'gauss_priors'):
            gausspriors=np.asarray(config.get('model', 'gauss_priors').split(','),dtype=np.float64)
        else:
            gausspriors=[]        
        
        bestfit_object=FindBestFit(min_prior, param_dict, low_data_arr, data_high=high_data_arr, print_fit=True, gausspriors=gausspriors)
        best_fit_result=bestfit_object.minimize_lowf_highf()
        best_fit_initial=np.array(best_fit_result['x'])
        init_dict=best_fit_to_dict(jnp.array(best_fit_initial), param_dict)
        init_dict_value=rescale_to_unit(init_dict, min_priors, max_priors)
        print(init_dict_value)
    
    samples=run_inference(model=model, min_priors=min_priors, max_priors=max_priors,
                        data_dict_1=low_data_arr, num_warmup=num_warmup,
                        num_samples=num_samples, max_tree_depth=max_tree_depth,
                        num_chains=num_chains, auto_corr=False, init_type=init_type, 
                        init_dict_value=init_dict_value, dense_mass=False, adapt_mass_matrix=True, 
                        target_accept_prob=target_accept_prob, kwargs=high_data_arr)
else:
    if init_type=="best_fit":
        param_dict=eval(config.get('model','model components'))
        min_prior=eval(config.get('model','min_bounds'))
        if config.has_option('model', 'gauss_priors'):
            gausspriors=np.asarray(config.get('model', 'gauss_priors').split(','),dtype=np.float64)
        else:
            gausspriors=[]

        bestfit_object=FindBestFit(min_prior, param_dict, low_data_arr, data_high=None, print_fit=True, gausspriors=gausspriors)
        best_fit_result=bestfit_object.minimize_lowf_highf()

        best_fit_initial=np.array(best_fit_result['x'])
        init_dict=best_fit_to_dict(jnp.array(best_fit_initial), param_dict)
        init_dict_value=rescale_to_unit(init_dict, min_priors, max_priors)
        print(init_dict_value)
        
    samples=run_inference(model=model, min_priors=min_priors, max_priors=max_priors,
                        data_dict_1=low_data_arr, num_warmup=num_warmup,
                        num_samples=num_samples, max_tree_depth=max_tree_depth,
                        num_chains=num_chains, auto_corr=False, init_type=init_type,
                        init_dict_value=init_dict_value, dense_mass=False, adapt_mass_matrix=True,
                        target_accept_prob=target_accept_prob)

with open(fname,"wb") as out_file:
    pickle.dump(samples,out_file)


  
  

  

