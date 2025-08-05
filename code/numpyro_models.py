import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax_sd_fg import * 
from numpyro_models_help import *

###################### models to sample with NUTS ##############################


################## dust with flat priors ##############################
def dT_y_dust_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_flat(prior_min, prior_max)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2, y_amp)+dust(x2, Td, Bd, Ad)
    
    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2", dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

def dT_y_dust(x, y, y_err, prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_flat(prior_min, prior_max)
    
    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)
    
    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

################## dust with T priors ##############################

def dT_y_dust_Tgauss(x, y, y_err, prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Tdgauss(prior_min, prior_max)
    
    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)
    
    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

def dT_y_dust_Tgauss_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):

    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Tdgauss(prior_min, prior_max)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2, y_amp)+dust(x2, Td, Bd, Ad)

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2", dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

################## baseline ##############################

def dT_y_dust_Bgauss(x, y, y_err, prior_min, prior_max):

    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)
    
    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)

    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

def dT_y_dust_Bgauss_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2,y_amp)+dust(x2,Td, Bd, Ad)

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2",dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

################## dust + extra A for high freqs ##############################

def dT_y_dust_Bgauss_joint_Aextra(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)
    
    Ahigh_unit=numpyro.sample('Ahigh_unit',dist.Uniform(low=-1, high=1))
    Ahigh_rescaled=rescale_uniform('Ahigh', Ahigh_unit, prior_min, prior_max)
    Ahigh=numpyro.deterministic('Ahigh',Ahigh_rescaled)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)
    model2=Ahigh*(dT(x2, DeltaT_amp)+DeltaI_y(x2,y_amp)+dust(x2,Td, Bd, Ad))

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2",dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)


##################### additional foregrounds: FF #################################
def dT_y_dust_Bgauss_FF_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)
    
    FF_amp_unit=numpyro.sample('EM_unit', dist.Uniform(low=-1, high=1))
    FF_amp_rescaled=rescale_uniform('EM', FF_amp_unit, prior_min, prior_max)
    EM=numpyro.deterministic('EM', FF_amp_rescaled)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)+jens_freefree_rad(x1,EM)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2,y_amp)+dust(x2,Td, Bd, Ad)+jens_freefree_rad(x2,EM)

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2",dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

def dT_y_dust_Bgauss_FF(x, y, y_err, prior_min, prior_max):
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)
    
    FF_amp_unit=numpyro.sample('EM_unit', dist.Uniform(low=-1, high=1))
    FF_amp_rescaled=rescale_uniform('EM', FF_amp_unit, prior_min, prior_max)
    EM=numpyro.deterministic('EM', FF_amp_rescaled)

    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)+jens_freefree_rad(x,EM)
    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

##################### additional foregrounds: fixed CIB #################################

def dT_y_dust_Bgauss_CIBfixed(x, y, y_err, prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    Acib_unit=numpyro.sample('Acib_unit',dist.Uniform(low=-1, high=1))
    Acib_rescaled=rescale_uniform('Acib', Acib_unit, prior_min, prior_max)
    Acib=numpyro.deterministic('Acib',Acib_rescaled)
   
    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)+cib_rad_MH23(x,Acib)
    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

def dT_y_dust_Bgauss_CIBfixed_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    Acib_unit=numpyro.sample('Acib_unit',dist.Uniform(low=-1, high=1))
    Acib_rescaled=rescale_uniform('Acib', Acib_unit, prior_min, prior_max)
    Acib=numpyro.deterministic('Acib',Acib_rescaled)
   
    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)+cib_rad_MH23(x1,Acib)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2,y_amp)+dust(x2,Td, Bd, Ad)+cib_rad_MH23(x2,Acib)

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2",dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

##################### additional foregrounds: CO #################################

def dT_y_dust_Bgauss_CO(x, y, y_err, prior_min, prior_max):
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    CO_amp_unit=numpyro.sample('CO_unit', dist.Uniform(low=-1, high=1))
    CO_amp_rescaled=rescale_uniform('CO', CO_amp_unit, prior_min, prior_max)
    CO=numpyro.deterministic('CO', CO_amp_rescaled)

    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)+co_rad(x,CO)
    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

def dT_y_dust_Bgauss_CO_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    CO_amp_unit=numpyro.sample('CO_unit', dist.Uniform(low=-1, high=1))
    CO_amp_rescaled=rescale_uniform('CO', CO_amp_unit, prior_min, prior_max)
    CO=numpyro.deterministic('CO', CO_amp_rescaled)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)+co_rad(x1,CO)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2,y_amp)+dust(x2,Td, Bd, Ad)+co_rad(x2,CO)

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2",dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

##################### additional foregrounds: CO+FF #################################

def dT_y_dust_Bgauss_CO_FF_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    CO_amp_unit=numpyro.sample('CO_unit', dist.Uniform(low=-1, high=1))
    CO_amp_rescaled=rescale_uniform('CO', CO_amp_unit, prior_min, prior_max)
    CO=numpyro.deterministic('CO', CO_amp_rescaled)

    FF_amp_unit=numpyro.sample('EM_unit', dist.Uniform(low=-1, high=1))
    FF_amp_rescaled=rescale_uniform('EM', FF_amp_unit, prior_min, prior_max)
    EM=numpyro.deterministic('EM', FF_amp_rescaled)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)+co_rad(x1,CO)+jens_freefree_rad(x1,EM)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2,y_amp)+dust(x2,Td, Bd, Ad)+co_rad(x2,CO)+jens_freefree_rad(x2,EM)

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2",dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

def dT_y_dust_Bgauss_CO_FF(x, y, y_err, prior_min, prior_max):
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    CO_amp_unit=numpyro.sample('CO_unit', dist.Uniform(low=-1, high=1))
    CO_amp_rescaled=rescale_uniform('CO', CO_amp_unit, prior_min, prior_max)
    CO=numpyro.deterministic('CO', CO_amp_rescaled)

    FF_amp_unit=numpyro.sample('EM_unit', dist.Uniform(low=-1, high=1))
    FF_amp_rescaled=rescale_uniform('EM', FF_amp_unit, prior_min, prior_max)
    EM=numpyro.deterministic('EM', FF_amp_rescaled)

    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)+co_rad(x,CO)+jens_freefree_rad(x,EM)
    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

##################### sync models #####################
def dT_y_dust_sync_Bdgauss(x, y, y_err,prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    #synch

    As, Bs=sample_sync_flat(prior_min, prior_max)

    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)+synch_rad(x, As, Bs)
    
    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)


def dT_y_dust_sync_Bdgauss_joint(x1, y1, y_err1,x2,y2,y_err2,prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    #synch

    As, Bs=sample_sync_flat(prior_min, prior_max)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)+synch_rad(x1, As, Bs)
    model2=dT(x2,DeltaT_amp)+DeltaI_y(x2,y_amp)+dust(x2,Td, Bd, Ad)+synch_rad(x2,As, Bs)
    
    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2", dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

def dT_y_dust_sync_Bdgauss_Bsgauss_joint(x1, y1, y_err1,x2,y2,y_err2,prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    #synch

    As, Bs=sample_sync_Bsgauss(prior_min, prior_max)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust(x1, Td, Bd, Ad)+synch_rad(x1, As, Bs)
    model2=dT(x2,DeltaT_amp)+DeltaI_y(x2,y_amp)+dust(x2,Td, Bd, Ad)+synch_rad(x2,As, Bs)

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2", dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)



def dT_y_dust_sync_Bdgauss_Bsgauss(x, y, y_err, prior_min, prior_max):

    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    #dust
    Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

    #synch

    As, Bs=sample_sync_Bsgauss(prior_min, prior_max)

    model=dT(x, DeltaT_amp)+DeltaI_y(x, y_amp)+dust(x, Td, Bd, Ad)+synch_rad(x, As, Bs)

    numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

################## relativistic ##################
# can't use analytic for high temperatures so functions below shouldn't be used unless priors are super tight
# def dT_y_rel_dust_Bdgauss(x, y, y_err, prior_min, prior_max):

#     #dT
#     DeltaT_amp=sample_dT(prior_min, prior_max)

#     #y
#     y_amp=sample_y(prior_min, prior_max)

#     #Trel
#     Trel_unit=numpyro.sample('Trel_unit', dist.Uniform(low=-1, high=1))
#     Trel_rescaled=rescale_uniform('Trel',Trel_unit,prior_min, prior_max)
#     Trel=numpyro.deterministic("Trel",  Trel_rescaled)

#     #dust
#     Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

#     model=dT(x, DeltaT_amp)+DeltaI_reltSZ(x, y_amp, Trel, prior_min['Yorder'])+dust(x, Td, Bd, Ad)

#     numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

# def dT_y_rel_dust_Bdgauss_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):

#     #dT
#     DeltaT_amp=sample_dT(prior_min, prior_max)

#     #y
#     y_amp=sample_y(prior_min, prior_max)

#     #Trel
#     Trel_unit=numpyro.sample('Trel_unit', dist.Uniform(low=-1, high=1))
#     Trel_rescaled=rescale_uniform('Trel',Trel_unit,prior_min, prior_max)
#     Trel=numpyro.deterministic("Trel",  Trel_rescaled)

#     #dust
#     Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

#     model1=dT(x1, DeltaT_amp)+DeltaI_reltSZ(x1, y_amp, Trel, prior_min['Yorder'])+dust(x1, Td, Bd, Ad)
#     model2=dT(x2, DeltaT_amp)+DeltaI_reltSZ(x2, y_amp, Trel, prior_min['Yorder'])+dust(x2, Td, Bd, Ad)

#     numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
#     numpyro.sample("y2", dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

# def dT_y_rel_dust_Bdgauss_SZpack(x, y, y_err, prior_min, prior_max):

#     #dT
#     DeltaT_amp=sample_dT(prior_min, prior_max)

#     #y
#     y_amp=sample_y(prior_min, prior_max)

#     #Trel
#     Trel_unit=numpyro.sample('Trel_unit', dist.Uniform(low=-1, high=1))
#     Trel_rescaled=rescale_uniform('Trel',Trel_unit,prior_min, prior_max)
#     Trel=numpyro.deterministic("Trel",  Trel_rescaled)

#     #dust
#     Td, Bd, Ad=sample_dust_Bdgauss(prior_min, prior_max)

#     model=dT(x, DeltaT_amp)+DeltaI_rel_SZpack(x, y_amp, Trel)+dust(x, Td, Bd, Ad)

#     numpyro.sample("y", dist.MultivariateNormal(loc=model/1.e5, covariance_matrix=y_err/(1.e5)**2), obs=y/1.e5)

####################### moments ###################################
def dT_y_dust_moments_omega2_omega3_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    
    #dust
    Ad, Td, Bd, omega2, omega3=sample_moments_first_order(prior_min, prior_max)

    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust_moments_omega2_omega3(x1, Ad, Td, Bd, omega2, omega3)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2, y_amp)+dust_moments_omega2_omega3(x2, Ad, Td, Bd, omega2, omega3)

    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2", dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

def dT_y_dust_moments_omega2_omega3_omega22_joint(x1, y1, y_err1, x2, y2, y_err2, prior_min, prior_max):
    
    #dT
    DeltaT_amp=sample_dT(prior_min, prior_max)
    #y
    y_amp=sample_y(prior_min, prior_max)
    
    #dust
    Ad, Td, Bd, omega2, omega3, omega22=sample_moments_first_order_omega22(prior_min, prior_max)
    
    model1=dT(x1, DeltaT_amp)+DeltaI_y(x1, y_amp)+dust_moments_omega2_omega3_omega22(x1,Ad, Td, Bd, omega2, omega3, omega22)
    model2=dT(x2, DeltaT_amp)+DeltaI_y(x2, y_amp)+dust_moments_omega2_omega3_omega22(x2,Ad, Td, Bd, omega2, omega3, omega22)
    
    numpyro.sample("y1", dist.MultivariateNormal(loc=model1/1.e5, covariance_matrix=y_err1/(1.e5)**2), obs=y1/1.e5)
    numpyro.sample("y2", dist.MultivariateNormal(loc=model2/1.e5, covariance_matrix=y_err2/(1.e5)**2), obs=y2/1.e5)

