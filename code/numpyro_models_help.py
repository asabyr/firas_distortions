import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_sd_fg import * 

#rescale from order unity to actual values
def rescale_normal(param, param_unit, prior_mu, prior_sigma):
    return param_unit*prior_sigma[param+'_sigma']+prior_mu[param+'_mu'] 
   
def rescale_uniform(param, param_unit, prior_min, prior_max):
    return (prior_max[param]-prior_min[param])/2.0*param_unit+(prior_max[param]+prior_min[param])/2.0

#help functions for sampling different sky components
def sample_dT(prior_min, prior_max):
    
    DeltaT_amp_unit=numpyro.sample('DeltaT_amp_unit', dist.Uniform(low=-1, high=1))   
    DeltaT_amp_rescaled=rescale_uniform('DeltaT_amp', DeltaT_amp_unit, prior_min, prior_max) 
    DeltaT_amp=numpyro.deterministic("DeltaT_amp", DeltaT_amp_rescaled)
    
    return DeltaT_amp

def sample_y(prior_min, prior_max):
    
    y_amp_unit=numpyro.sample('y_amp_unit', dist.Uniform(low=-1, high=1))
    y_amp_rescaled=rescale_uniform('y_amp', y_amp_unit, prior_min, prior_max) 
    y_amp=numpyro.deterministic("y_amp",  y_amp_rescaled)
    
    return y_amp

def sample_dust_flat(prior_min, prior_max):
    
    Td_unit=numpyro.sample('Td_unit', dist.Uniform(low=-1, high=1))
    Bd_unit=numpyro.sample('Bd_unit', dist.Uniform(low=-1, high=1))
    Ad_unit=numpyro.sample('Ad_unit', dist.Uniform(low=-1, high=1))
    
    Td_rescaled=rescale_uniform('Td', Td_unit, prior_min, prior_max) 
    Bd_rescaled=rescale_uniform('Bd', Bd_unit, prior_min, prior_max) 
    Ad_rescaled=rescale_uniform('Ad', Ad_unit, prior_min, prior_max) 
    
    Td=numpyro.deterministic('Td', Td_rescaled)
    Bd=numpyro.deterministic('Bd', Bd_rescaled)
    Ad=numpyro.deterministic('Ad', Ad_rescaled)

    return Td, Bd, Ad

def sample_dust_Bdgauss(prior_min, prior_max):

    Bd_unit=numpyro.sample('Bd_unit', dist.Normal(loc=0.0, scale=1))
    Td_unit=numpyro.sample('Td_unit', dist.Uniform(low=-1, high=1))
    Ad_unit=numpyro.sample('Ad_unit', dist.Uniform(low=-1, high=1))

    Td_rescaled=rescale_uniform('Td', Td_unit, prior_min, prior_max) 
    Bd_rescaled=rescale_normal('Bd', Bd_unit, prior_min, prior_max) 
    Ad_rescaled=rescale_uniform('Ad', Ad_unit, prior_min, prior_max) 

    Td=numpyro.deterministic('Td', Td_rescaled)
    Bd=numpyro.deterministic('Bd', Bd_rescaled)
    Ad=numpyro.deterministic('Ad', Ad_rescaled)

    return Td, Bd, Ad

def sample_dust_Tdgauss(prior_min, prior_max):

    Td_unit=numpyro.sample('Bd_unit', dist.Normal(loc=0.0, scale=1))
    Bd_unit=numpyro.sample('Td_unit', dist.Uniform(low=-1, high=1))
    Ad_unit=numpyro.sample('Ad_unit', dist.Uniform(low=-1, high=1))

    Bd_rescaled=rescale_uniform('Bd', Bd_unit, prior_min, prior_max) 
    Td_rescaled=rescale_normal('Td', Td_unit, prior_min, prior_max) 
    Ad_rescaled=rescale_uniform('Ad', Ad_unit, prior_min, prior_max) 

    Td=numpyro.deterministic('Td', Td_rescaled)
    Bd=numpyro.deterministic('Bd', Bd_rescaled)
    Ad=numpyro.deterministic('Ad', Ad_rescaled)

    return Td, Bd, Ad

def sample_sync_flat(prior_min, prior_max):
    
    As_unit=numpyro.sample('As_unit', dist.Uniform(low=-1, high=1))
    Bs_unit=numpyro.sample('Bs_unit', dist.Uniform(low=-1, high=1))

    As_rescaled=rescale_uniform('As', As_unit, prior_min, prior_max)
    Bs_rescaled=rescale_uniform('Bs', Bs_unit, prior_min, prior_max)

    As=numpyro.deterministic('As', As_rescaled)
    Bs=numpyro.deterministic('Bs', Bs_rescaled)

    return As, Bs

def sample_sync_Bsgauss(prior_min, prior_max):
    As_unit=numpyro.sample('As_unit', dist.Uniform(low=-1, high=1))
    Bs_unit=numpyro.sample('Bs_unit', dist.Normal(loc=0.0, scale=1))

    As_rescaled=rescale_uniform('As', As_unit, prior_min, prior_max)
    Bs_rescaled=rescale_normal('Bs', Bs_unit, prior_min, prior_max)

    As=numpyro.deterministic('As', As_rescaled)
    Bs=numpyro.deterministic('Bs', Bs_rescaled)

    return As, Bs

def sample_moments_first_order(prior_min, prior_max):

    Ad_unit=numpyro.sample('Ad_unit', dist.Uniform(low=-1, high=1))
    omega2_unit=numpyro.sample('omega2_unit', dist.Uniform(low=-1, high=1))
    omega3_unit=numpyro.sample('omega3_unit', dist.Uniform(low=-1, high=1))

    Ad_rescaled=rescale_uniform('Ad', Ad_unit, prior_min, prior_max)
    omega2_rescaled=rescale_uniform('omega2', omega2_unit, prior_min, prior_max)
    omega3_rescaled=rescale_uniform('omega3', omega3_unit, prior_min, prior_max)

    Td=prior_min['Td_fix']
    Bd=prior_min['Bd_fix']

    Ad=numpyro.deterministic('Ad', Ad_rescaled)
    omega2=numpyro.deterministic('omega2', omega2_rescaled)
    omega3=numpyro.deterministic('omega3', omega3_rescaled)

    return Ad, Td, Bd, omega2, omega3

def sample_moments_first_order_omega22(prior_min, prior_max):

    Ad_unit=numpyro.sample('Ad_unit', dist.Uniform(low=-1, high=1))
    omega2_unit=numpyro.sample('omega2_unit', dist.Uniform(low=-1, high=1))
    omega3_unit=numpyro.sample('omega3_unit', dist.Uniform(low=-1, high=1))
    omega22_unit=numpyro.sample('omega22_unit', dist.Uniform(low=-1, high=1))
    
    Ad_rescaled=rescale_uniform('Ad', Ad_unit, prior_min, prior_max)
    omega2_rescaled=rescale_uniform('omega2', omega2_unit, prior_min, prior_max)
    omega3_rescaled=rescale_uniform('omega3', omega3_unit, prior_min, prior_max)
    omega22_rescaled=rescale_uniform('omega22', omega22_unit, prior_min, prior_max)

    Td=prior_min['Td_fix']
    Bd=prior_min['Bd_fix']
    
    Ad=numpyro.deterministic('Ad', Ad_rescaled)
    omega2=numpyro.deterministic('omega2', omega2_rescaled)
    omega3=numpyro.deterministic('omega3', omega3_rescaled)
    omega22=numpyro.deterministic('omega22', omega22_rescaled)
    
    return Ad, Td, Bd, omega2, omega3, omega22
