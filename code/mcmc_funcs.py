import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import h5py

file_path=os.path.dirname(os.path.abspath(__file__))
dir_path=file_path.replace('mcmc_funcs','')
sys.path.append(dir_path)

from minimization_funcs import *
from priors import *
from Inu import *

# Set the path to the desired emcee version
sys.path.append('/moto/home/as6131/.conda/envs/firas/lib/python3.9/site-packages')
import emcee
from multiprocessing import Pool

#logL functions for low only and low+high frequencies
# theta -- free param array,
# total_Inu-- sky model function from Inu.py
# cov_inv_1, data_nu_1, data_Inu_1 -- inverse covariance, freqs, intensity

def logL_joint(theta,total_Inu,cov_inv_1, data_nu_1, data_Inu_1,
               cov_inv_2, data_nu_2, data_Inu_2):

  model_1=total_Inu(data_nu_1,theta)
  delta_x_1=data_Inu_1-model_1
  step1_1=np.dot(delta_x_1.T,cov_inv_1)
  step2_1=np.dot(step1_1,delta_x_1)

  model_2=total_Inu(data_nu_2,theta)
  delta_x_2=data_Inu_2-model_2
  step1_2=np.dot(delta_x_2.T,cov_inv_2)
  step2_2=np.dot(step1_2,delta_x_2)

  if np.isfinite(step2_1)==False or np.isfinite(step2_2)==False:

      return -np.inf

  return -0.5*(step2_1+step2_2)

def logL(theta,total_Inu,cov_inv_1, data_nu_1, data_Inu_1):

  model_1=total_Inu(data_nu_1,theta)
  delta_x_1=data_Inu_1-model_1
  step1_1=np.dot(delta_x_1.T,cov_inv_1)
  step2_1=np.dot(step1_1,delta_x_1)

  if np.isfinite(step2_1)==False:

      return -np.inf

  return -0.5*(step2_1)


#log probability, prior is a prior function from priors.py
def logProb_joint(theta, total_Inu, prior, cov_inv_1, data_nu_1, data_Inu_1,
                  cov_inv_2, data_nu_2, data_Inu_2):

    lprior=prior(theta)

    if np.isfinite(lprior)==False:
        return -np.inf

    return lprior+logL_joint(theta,total_Inu, cov_inv_1, data_nu_1, data_Inu_1, cov_inv_2,data_nu_2, data_Inu_2)

def logProb(theta, total_Inu,prior, cov_inv_1, data_nu_1, data_Inu_1):

    lprior=prior(theta)

    if np.isfinite(lprior)==False:
        return -np.inf

    return lprior+logL(theta,total_Inu,cov_inv_1, data_nu_1, data_Inu_1)

def MCMC_run(prior,
             total_Inu,
             params_names,
            mcmc_dict,
            out_fname,
            data_dict_1,
            cont_run=True,
            diff_moves=[],
            parallel=True,
            show_progress=False,
            best_fit=[],*args):

    """
    Function runs emcee.
    Input:
    prior --prior function from priors.py
    total_Inu -- Inu function from Inu.py
    params_names (dict) -- sky components
    mcmc_dict (dict) -- mcmc settings dictionary
    out_fname (str) -- file name to save samples to
    data_dict_1 (dict) -- data including x,y,covariance

    cont_run (bool) -- continue from out_fname.h5 file or not
    diff_moves (float array) -- fractional moves (DE and DESnooker)
    parallel (bool) -- run in parallel or not
    show_progress (bool) -- use tqdm to show progress

    best_fit (float array) -- initialize from these parameter values
    *args (dict) -- can input high frequency data dict here

    Output: None
    """


    #mcmc settings
    walkers=mcmc_dict['n walkers']
    mcmc_steps=mcmc_dict['steps']
    dtheta=mcmc_dict['initial param perturb [%]']

    #initial values: either best fit or fiducial from Abitbol+17
    if len(best_fit)>1:
        initial_theta=best_fit.copy()
    else:
        initial_theta=params_initial_Abitbol2017(params_names)

    pos = initial_theta + initial_theta*dtheta * np.random.randn(walkers, len(initial_theta))
    nwalkers, ndim = pos.shape

    #low frequency data
    freqs_1=data_dict_1['freqs']
    firas_SD_fg_1=data_dict_1['intensity']
    cov_inv_1=data_dict_1['cov_inv']

    #initialize backend
    backend = emcee.backends.HDFBackend(f"{out_fname}.h5")

    #set positions to None if continuing the runs
    if cont_run==False:
        backend.reset(nwalkers, ndim)
    else:
        pos = None

    #check extra argument
    if len(args)>1:
        sys.exit("only one extra argument allowed: second data dictionary")

    #run default emcee
    elif len(args)==1 and len(diff_moves)<1 and parallel==False:
        print("running both lowf and highf")

        freqs_2=args[0]['freqs']
        firas_SD_fg_2=args[0]['intensity']
        cov_inv_2=args[0]['cov_inv']

        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb_joint, args=(total_Inu,prior,cov_inv_1, freqs_1, firas_SD_fg_1,cov_inv_2, freqs_2, firas_SD_fg_2),backend=backend)
        sampler.run_mcmc(pos, mcmc_steps, progress=show_progress);

    elif len(args)==0 and len(diff_moves)<1 and parallel==False:
        print("running lowf")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb, args=(total_Inu,prior, cov_inv_1, freqs_1, firas_SD_fg_1),backend=backend)
        sampler.run_mcmc(pos, mcmc_steps, progress=show_progress);

    ######## options below are not used but keeping them in case useful later ######## 
    #run using different moves (DE and DESnooker)
    elif len(args)==1 and len(diff_moves)>1 and parallel==False:
        print("running both lowf and highf and combination of moves")
        freqs_2=args[0]['freqs']
        firas_SD_fg_2=args[0]['intensity']
        cov_inv_2=args[0]['cov_inv']

        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb_joint,moves=[(emcee.moves.DEMove(), diff_moves[0]),(emcee.moves.DESnookerMove(),diff_moves[1])], args=(total_Inu,prior,cov_inv_1, freqs_1, firas_SD_fg_1,cov_inv_2, freqs_2, firas_SD_fg_2),backend=backend)
        sampler.run_mcmc(pos, mcmc_steps, progress=show_progress);

    elif len(args)==0 and len(diff_moves)>1 and parallel==False:
        print("running lowf and combination of moves")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb,moves=[(emcee.moves.DEMove(), diff_moves[0]),(emcee.moves.DESnookerMove(),diff_moves[1])], args=(total_Inu,prior, cov_inv_1, freqs_1, firas_SD_fg_1),backend=backend)
        sampler.run_mcmc(pos, mcmc_steps, progress=show_progress);

    #run in parallel
    elif len(args)==1 and len(diff_moves)<1 and parallel==True:
        os.environ["OMP_NUM_THREADS"] = "1"
        print("running in parallel both lowf and highf")
        freqs_2=args[0]['freqs']
        firas_SD_fg_2=args[0]['intensity']
        cov_inv_2=args[0]['cov_inv']
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb_joint, args=(total_Inu,prior,cov_inv_1, freqs_1, firas_SD_fg_1,cov_inv_2, freqs_2, firas_SD_fg_2),backend=backend, pool=pool)
            sampler.run_mcmc(pos, mcmc_steps, progress=show_progress);

    elif len(args)==0 and len(diff_moves)<1 and parallel==True:
        print("running in parallel lowf")
        os.environ["OMP_NUM_THREADS"] = "1"
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb, args=(total_Inu,prior, cov_inv_1, freqs_1, firas_SD_fg_1),backend=backend, pool=pool)
            sampler.run_mcmc(pos, mcmc_steps, progress=show_progress);


    #autocorrelation time & acceptance fraction
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"Mean autocorrelation time={tau}")
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
