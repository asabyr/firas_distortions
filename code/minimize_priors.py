import numpy as np
import sys
sys.path.append('/Users/asabyr/Documents/software/sd_foregrounds')
sys.path.append('/Users/asabyr/Documents/firas_distortions/code')
from spectral_distortions import *
from foregrounds import *

###########prior bounds for scipy optimize########
#probably a better way to do this, but saved here to avoid errors
#baseline
def params_priors_dust_gaussBd(params_names):
    x0=[]
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append((None, None))
        elif param==DeltaI_y:
            x0.append((None, None))
        elif param==thermal_dust_rad:
            x0.append((None, None))
            x0.append((None, None))
            x0.append((0, 100))
        elif param==DeltaI_rel_SZpack:#old run
            x0.append((None, None))
            x0.append((1e-10,70)) #SZpack doesn't compute for kTe=0
    return tuple(x0)

#baseline for rel. run
def params_priors_dust_gaussBd_rel_0to190(params_names):
    x0=[]
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append((None, None))
        elif param==DeltaI_y:
            x0.append((None, None))
        elif param==thermal_dust_rad:
            x0.append((None, None))
            x0.append((None, None))
            x0.append((0, 100))
        elif param==DeltaI_rel_SZpack:
            x0.append((0, None))
            x0.append((1e-10,190)) #SZpack doesn't compute for kTe=0
    return tuple(x0)


def params_priors_dust_gaussTd(params_names):
    x0=[]
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append((None, None))
        elif param==DeltaI_y:
            x0.append((None, None))
        elif param==thermal_dust_rad:
            x0.append((None, None))
            x0.append((0,3))
            x0.append((None,None))
    return tuple(x0)

def params_priors_dust_flat(params_names):
    x0=[]
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append((None, None))
        elif param==DeltaI_y:
            x0.append((None, None))
        elif param==thermal_dust_rad:
            x0.append((None, None))
            x0.append((0,3))
            x0.append((0, 100))
    return tuple(x0)

#extra foregrounds
#baseline freqs
def params_priors_dust_Bd_other_FGs(params_names):
    x0=[]
    
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append((None, None))
        elif param==DeltaI_y:
            x0.append((None, None))
        elif param==thermal_dust_rad:
            x0.append((None, None))
            x0.append((None, None))
            x0.append((0, 100))
        elif param==co_rad:
            x0.append((None, None))
        elif param==jens_freefree_rad:
            x0.append((None, None))
        elif param==jens_synch_rad_no_curv:
            x0.append((None, None))
            x0.append((None, None))
        # elif param==jens_synch_rad:
        #     x0.append((None, None))
        #     x0.append((None, None))
        #     x0.append((None, None))
        elif param==cib_rad_MH23:
            x0.append((None, None))
        # elif param==cib_rad_A17:
        #     x0.append((None, None))
        # elif param==cib_rad:
        #     x0.append((None, None))
        #     x0.append((None, None))
        #     x0.append((None, None))
        # elif param==DeltaI_reltSZ: #not used anymore
        #     x0.append((1.87e-7, None))
        #     x0.append((-1000,1000))
        # elif param==DeltaI_reltSZ_Y3:
        #     x0.append((1.87e-7, None))
        #     x0.append((-1000,1000))
        # elif param==DeltaI_reltSZ_Y1:
        #     x0.append((1.87e-7, None))
        #     x0.append((-1000,1000))
        # elif param==DeltaI_reltSZ_Y2:
        #     x0.append((1.87e-7, None))
        #     x0.append((-1000,1000))

    return tuple(x0)

#high freqs appendix
def params_priors_dust_flat_other_FGs(params_names):
    x0=[]
    
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append((None, None))
        elif param==DeltaI_y:
            x0.append((None, None))
        elif param==thermal_dust_rad:
            x0.append((None, None))
            x0.append((0, 3))
            x0.append((0, 100))
        elif param==co_rad:
            x0.append((None, None))
        elif param==jens_freefree_rad:
            x0.append((None, None))
        elif param==jens_synch_rad_no_curv:
            x0.append((None, None))
            x0.append((None, None))
        elif param==jens_synch_rad:
            x0.append((None, None))
            x0.append((None, None))
            x0.append((None, None))
        elif param==powerlaw:
            x0.append((None, None))
            x0.append((None, None))
        elif param==cib_rad_MH23:
            x0.append((None, None))
        # elif param==cib_rad_A17:
        #     x0.append((None, None))
        # elif param==cib_rad:
        #     x0.append((None, None))
        #     x0.append((None, None))
        #     x0.append((None, None))
        # elif param==DeltaI_reltSZ:#not used anymore
        #     x0.append((1.87e-7, None))
        #     x0.append((-1000,1000))
    return tuple(x0)

def params_priors_dust_flat_sync_flat_other_FGs(params_names):
    x0=[]
    
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append((None, None))
        elif param==DeltaI_y:
            x0.append((None, None))
        elif param==thermal_dust_rad:
            x0.append((None, None))
            x0.append((0, 3))
            x0.append((0, 100))
        elif param==co_rad:
            x0.append((None, None))
        elif param==jens_freefree_rad:
            x0.append((None, None))
        elif param==jens_synch_rad_no_curv:
            x0.append((None, None))
            x0.append((-5,5))
        elif param==jens_synch_rad:
            x0.append((None, None))
            x0.append((-5,5))
            x0.append((None, None))
        elif param==powerlaw:
            x0.append((None, None))
            x0.append((None, None))
        # elif param==cib_rad_MH23:
        #     x0.append((None, None))
        # elif param==cib_rad_A17:
        #     x0.append((None, None))
        # elif param==cib_rad:
        #     x0.append((None, None))
        #     x0.append((None, None))
        #     x0.append((None, None))
        # elif param==DeltaI_reltSZ:
        #     x0.append((1.87e-7, None))
        #     x0.append((-1000,1000))
    return tuple(x0)

def params_priors_moments(params_names):
    x0=[]
    
    for param in params_names:
        if param==DeltaI_DeltaT:
            x0.append((None, None))
        elif param==DeltaI_y:
            x0.append((None, None))
        elif param==dust_moments_omega2_omega3:
            x0.append((None, None))
            x0.append((-1000, 1000))
            x0.append((-1000, 1000))
        elif param==dust_moments_omega2_omega3_bestfit:
            x0.append((None, None))
            x0.append((-1000, 1000))
            x0.append((-1000, 1000))
        elif param==dust_moments_omega2_omega3_omega22:
            x0.append((None, None))
            x0.append((-1000, 1000))
            x0.append((-1000, 1000))
            x0.append((-1000, 1000))
        # elif param==dust_moments_omega2_omega3_omega22_omega33:
        #     x0.append((None, None))
        #     x0.append((-1000, 1000))
        #     x0.append((-1000, 1000))
        #     x0.append((-1000, 1000))
        #     x0.append((-1000, 1000))
    return tuple(x0)
