import numpy as np
from help_funcs import *
from getdist import MCSamples
import constants as const

class read_NUTS_posteriors:
    
    def __init__(self,fname, param_dict, priors, burn_in_frac=0.3, return_upper=False):
        """class to read pkl posteriors and quickly return mean, 
        sigmas and a parameter dictionary (for fisher)
        
        """
        self.fname=fname #file, full path
        self.param_dict=param_dict #sky model dictionary
        self.burn_in_frac=burn_in_frac #burn in, in terms of fraction
        self.clean_samples() 
        self.return_upper=return_upper
        self.priors=priors

    def clean_samples(self):
        
        #make an array from pkl file and throw away burn in
        self.parameters=nuts_params(self.param_dict)
        nuts_samples=np.load(self.fname, allow_pickle=True)
        
        n_params=len(self.parameters)
        chains, steps=np.shape(nuts_samples[self.parameters[0]])
        burn_in=int(self.burn_in_frac*steps)

        n_samples=int(chains*(steps-burn_in))
        clean_steps=int(steps-burn_in)

        np_nuts_samples=np.zeros((chains,clean_steps,n_params))
        
        for i in range(len(self.parameters)):
            #scale to physical quantities 
            if self.parameters[i]=='DeltaT_amp' or self.parameters[i]=='y_amp':
                # oneparam=nuts_samples[self.parameters[i]]
                np_nuts_samples[:,:,i]=nuts_samples[self.parameters[i]][:,burn_in:]/const.dT_factor
            else:
                np_nuts_samples[:,:,i]=nuts_samples[self.parameters[i]][:,burn_in:]
    
        
        # burn_in=int(self.burn_in_frac*len(np_nuts_samples[:,0]))
        self.newsamples=np.reshape(np_nuts_samples, (n_samples, n_params))
    
    
    def read_mean(self):
        return np.mean(self.newsamples, axis=0)
    
    def return_sigma(self):
        #return parameters, mean, and their sigma
        prior_keys=np.array(list(self.priors.keys()))
        assert np.array_equal(prior_keys, self.parameters)==True, "check prior names"

        samples_getdist = MCSamples(samples=self.newsamples, names=self.parameters, ranges=self.priors)
        # print(samples_getdist.getInlineLatex('y_amp',limit=1, err_sig_figs=5)) #print 68% confidence


        sigmas=[]
        upper_lims=[]
        means=[]
        j=0
        for param in self.parameters:
            
            stats = samples_getdist.getMargeStats()
            lims1 = stats.parWithName(param).limits

            # param_mean=np.mean(self.newsamples, axis=0)
            param_mean=stats.parWithName(param).mean
            # print(lims1[0].lower-param_mean[j])
            # print(lims1[1].lower-param_mean[j])
            # print(stats.parWithName(param).err)
            
            st_dev_min=lims1[0].lower-param_mean
            st_dev_plus=lims1[0].upper-param_mean
        
            if param=='y_amp':
                # print(st_dev_min*1e6)
                # print(st_dev_plus*1e6)
                sigmas.append([st_dev_min*1e6,st_dev_plus*1e6])
                upper_lims.append(lims1[1].upper*1e6)
                means.append(param_mean)
            else:
                sigmas.append([st_dev_min,st_dev_plus])
                upper_lims.append(lims1[1].upper)
                means.append(param_mean)
            j+=1

        if self.return_upper==True:
              return self.parameters, means, sigmas, upper_lims
        return self.parameters, means, sigmas
    
    def return_dict(self):
        #make a dictionary that can be used in Fisher forecasts
        prior_keys=np.array(list(self.priors.keys()))
        assert np.array_equal(prior_keys, self.parameters)==True, "check prior names"
        
        samples_getdist = MCSamples(samples=self.newsamples, names=self.parameters, ranges=self.priors)

        arg_dict={}
        for param in self.parameters:
            stats = samples_getdist.getMargeStats()
            param_mean=stats.parWithName(param).mean
            if param=='y_amp':
                arg_dict['y_tot']=param_mean
            else:
                arg_dict[param]=param_mean
        return arg_dict
        