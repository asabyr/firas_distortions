#project modules
import sys
sys.path.append('/moto/home/as6131/firas_distortions/')
sys.path.append('/moto/hill/users/as6131/software/sd_foregrounds/')
from read_data import *
from help_funcs import *
import constants as const
from spectral_distortions import *
from foregrounds import *

#numpyro things
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary, autocorrelation, autocovariance
from numpyro.infer import MCMC, NUTS
import arviz
import matplotlib.pyplot as plt
###very important to set this to True, will fail otherwise in many cases
numpyro.util.enable_x64(use_x64=True)

#NUTs is very sensitive to covariance being exactly symmetric
#throws "invalid" error even if symmetric to many digits
def fix_cov(cov):
    cov_new=np.copy(cov)
    np.fill_diagonal(cov_new, np.zeros(len(np.diag(cov))))
    fixed_cov = (np.tril(cov_new) + np.triu(cov_new).T)/2+(np.tril(cov_new).T + np.triu(cov_new))/2
    np.fill_diagonal(fixed_cov, np.diag(cov))
    return fixed_cov

def run_inference(model, min_priors, max_priors,
                  data_dict_1,
                  num_warmup=1000,
                  num_samples=1000,
                  max_tree_depth=10,
                  num_chains=4, auto_corr=False, init_type='best_fit',
                init_dict_value={}, dense_mass=False,
                adapt_mass_matrix=False, target_accept_prob=0.8, **kwargs):
    """function to run NUTs. See https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS.
    Input:

    model -- model function from numpyro_models.py
    min_priors -- dictionary with minimum prior bound
    max_prior -- dictionary with maximum prior bound
    data_dict_1 -- data dictionary including covariance, frequencies and intensity

    num_warmup -- warmup steps (often the same as the number of mcmc steps)
    num_sampels -- mcmc steps
    max_tree_depth -- default is 10 in numpyro (15 is usually enough for firas data)
    num_chains -- need at least 4 (usually set to 10)

    *not really varied, just experimented with at the beginning*
    auto_corr -- print auto correlation
    dense_mass -- False (mass matrix is full rank or diagonal, keep diagonal)
    adapt_mass_matrix -- should be set to True (adapt during warmup, makes a difference)
    target_accept_prob -- kept at default 0.8

    init_type -- initialization strategy ('feasible' seems to work well)
    init_dict_value -- initial values (optional)

    **kwargs -- high frequency data dictionary

    """
    if len(kwargs)>1:
        sys.exit("only one extra argument allowed: second data dictionary")

    #different initializations
    if init_type=='best_fit' or init_type=='custom':

        kernel = NUTS(model,
                      adapt_step_size=True,
                      max_tree_depth=max_tree_depth,target_accept_prob=target_accept_prob,
                      init_strategy=numpyro.infer.initialization.init_to_value(values=init_dict_value), dense_mass=dense_mass,adapt_mass_matrix=adapt_mass_matrix)
    elif init_type=='median':

        kernel = NUTS(model,
                      adapt_step_size=True,
                      max_tree_depth=max_tree_depth,target_accept_prob=target_accept_prob,
                      init_strategy=numpyro.infer.initialization.init_to_median, dense_mass=dense_mass,adapt_mass_matrix=adapt_mass_matrix)
    #this is the one that's used in the paper
    elif init_type=='feasible':

        kernel = NUTS(model,
                      adapt_step_size=True,
                      max_tree_depth=max_tree_depth,target_accept_prob=target_accept_prob,
                      init_strategy=numpyro.infer.initialization.init_to_feasible, dense_mass=dense_mass,adapt_mass_matrix=adapt_mass_matrix)

    ## Define MCMC routine
    mcmc = MCMC(kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    progress_bar=True)

    #low freqs
    x_data_1=data_dict_1['freqs']
    y_data_1=data_dict_1['intensity']
    y_err_data_1=fix_cov(data_dict_1['cov'])
    

    if len(kwargs)==1:
        #high frequency data included
        data_dict_2=kwargs['kwargs']
        x_data_2=data_dict_2['freqs']
        y_data_2=data_dict_2['intensity']
        y_err_data_2=fix_cov(data_dict_2['cov'])

        #validation_enabled helps debug
        # with numpyro.validation_enabled():
        mcmc.run(random.PRNGKey(1), x1=x_data_1, y1=y_data_1, y_err1=y_err_data_1, x2=x_data_2, y2=y_data_2, y_err2=y_err_data_2, prior_min=min_priors, prior_max=max_priors)

    else:

        #with numpyro.validation_enabled():
        mcmc.run(random.PRNGKey(1), x=x_data_1, y=y_data_1, y_err=y_err_data_1, prior_min=min_priors, prior_max=max_priors)

    mcmc.print_summary()

    #print autocorrelation times
    if auto_corr==True:
        for ii, jj in mcmc.get_samples().items():
            print(jj)
            print(f"aucorr: {ii}")
            print(autocorrelation(jj, axis=0))

    #suggested in Dan's blog post on numpyro
    # import arviz as az
    inf_data = az.from_numpyro(mcmc)
    print(az.summary(inf_data))

    return mcmc.get_samples(group_by_chain=True)

#take some initial parameters & prior bounds and rescale to unity
#for testing initialization 
def rescale_to_unit(param_dict, min_prior, max_prior):

    rescaled={}

    for key, value in param_dict.items():
        if key+'_mu' in min_prior.keys() and key+'_sigma' in max_prior.keys() :
            rescaled[key+'_unit']=(value-min_prior[key+'_mu'])/max_prior[key+'_sigma']
        else:
            rescaled[key+'_unit']=(value-(max_prior[key]+min_prior[key])/2.0)/((max_prior[key]-min_prior[key])/2.0)

    return rescaled

#turn best fit array into a best fit dictionary
def best_fit_to_dict(best, param_dict):

    best_keys=nuts_params(param_dict)
    best_dict={}
    for i in range(len(best)):
        best_dict[best_keys[i]]=best[i]

    return best_dict


###################used for quick plotting (not in the paper) ###################
def plot_nuts_samples(nuts_samples, parameters, labels, priors, burn_in=0.3,save_fig=False, save_samples=False, name=''):

    n_params=len(parameters)
    n_samples=len(nuts_samples[parameters[0]])
    np_nuts_samples=np.zeros((n_samples, n_params))

    for i in range(len(parameters)):

        if parameters[i]=='DeltaT_amp' or parameters[i]=='y_amp':
            np_nuts_samples[:,i]=nuts_samples[parameters[i]]/const.dT_factor
        else:
            np_nuts_samples[:,i]=nuts_samples[parameters[i]]

    from getdist import plots, MCSamples
    import getdist
    burn_in_steps=int(burn_in*len(np_nuts_samples[:,0]))

    prior_keys=np.array(list(priors.keys()))
    assert np.array_equal(prior_keys, labels)==True, "check prior names"
        
    nuts_samples1 = MCSamples(samples=np_nuts_samples[burn_in_steps:,:], names = labels, labels = labels, ranges=priors)

    plt.ion()
    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.4
    g.settings.title_limit_fontsize = 14

    fig=g.triangle_plot([nuts_samples1],
            filled=True,
            legend_loc='upper right',
            title_limit=1)

    if save_fig==True:
        plt.savefig("../mcmc_results_NUTs/"+name+".pdf")
    if save_samples==True:
        np.savetxt("../mcmc_results_NUTs/"+name+".txt", np_nuts_samples)

def plot_nuts_file(fname, parameters, labels, priors, save_fig=False, save_samples=False, name='',
                   burn_in=0.3, colors=['blue'], keep=5, throw=[], eff_sample=1000, r_hat_max=1.01):


    nuts_samples=np.load(fname, allow_pickle=True)
    a_summary=arviz.summary(nuts_samples)
    print(a_summary)
    if np.any(a_summary['ess_bulk'].values<eff_sample)==False and np.any(a_summary['r_hat'].values>r_hat_max)==False:

        n_params=len(parameters)
        chains, steps=np.shape(nuts_samples[parameters[0]])

        #in the end, I don't throw out chains, this is just for testing
        keep_chains=[]
        i=0
        while len(keep_chains)<keep:
            if i not in throw:
                keep_chains.append(i)
            i+=1
        keep_chains=np.array(keep_chains)
        # print(keep_chains)
        burn_in_steps=int(burn_in*steps)
        clean_steps=int(steps-burn_in_steps)
        n_samples=int(keep*clean_steps)

        np_nuts_samples=np.zeros((keep, clean_steps, n_params))

        for i in range(len(parameters)):

            if parameters[i]=='DeltaT_amp' or parameters[i]=='y_amp':
                np_nuts_samples[:,:,i]=nuts_samples[parameters[i]][keep_chains, burn_in_steps:]/const.dT_factor
            else:

                np_nuts_samples[:,:,i]=nuts_samples[parameters[i]][keep_chains, burn_in_steps:]
        print(np.shape(np_nuts_samples))
        from getdist import plots, MCSamples
        import getdist
        newsamples=np.reshape(np_nuts_samples, (n_samples, n_params))
        
        prior_keys=np.array(list(priors.keys()))
        assert np.array_equal(prior_keys, parameters)==True, "check prior names"
    
        nuts_samples1 = MCSamples(samples=newsamples, names = parameters, labels = labels, ranges=priors)

        plt.ion()
        g = plots.get_subplot_plotter()
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.4
        g.settings.title_limit_fontsize = 14

        fig=g.triangle_plot([nuts_samples1],
                filled=True,
                legend_loc='upper right',
                title_limit=1,contour_colors=colors)

        if save_fig==True:
            plt.savefig("../mcmc_results_NUTs/"+name+".pdf")
        if save_samples==True:
            np.savetxt("../mcmc_results_NUTs/"+name+".txt", np_nuts_samples)

        return np.mean(newsamples, axis=0)
    else:
        sys.exit("NUTS may not have converged...")
