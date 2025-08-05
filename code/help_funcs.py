import sys
sys.path.append('/Users/asabyr/Documents/software/sd_foregrounds')
sys.path.append('/Users/asabyr/Documents/firas_distortions/code')
from spectral_distortions import *
from foregrounds import *
import numpy as np
import constants as const

#####################
#nuts parameter labels for getdist
def nuts_params(param_dict):

    all_params=[]
    for param in param_dict:
        if param==DeltaI_DeltaT:
            all_params.append('DeltaT_amp')
        if param==DeltaI_y:
            all_params.append('y_amp')
        if param==DeltaI_reltSZ: #not used
            all_params.append('y_amp')
            all_params.append('Trel')
        if param==DeltaI_reltSZ_Y3: #not used
            all_params.append('y_amp')
            all_params.append('Trel')
        if param==DeltaI_reltSZ_Y2: #not used
            all_params.append('y_amp')
            all_params.append('Trel')
        if param==DeltaI_reltSZ_Y1: #not used
            all_params.append('y_amp')
            all_params.append('Trel')
        if param==thermal_dust_rad:
            all_params.append('Ad')
            all_params.append('Bd')
            all_params.append('Td')
        if param==jens_freefree_rad:
            all_params.append('EM')
        if param==co_rad:
            all_params.append('CO')
        if param==jens_synch_rad_no_curv:
            all_params.append('As')
            all_params.append('Bs')
        if param==jens_synch_rad:
            all_params.append('As')
            all_params.append('Bs')
            all_params.append('omega')
        if param==dust_moments_omega2_omega3_omega22:
            all_params.append('Ad')
            all_params.append('omega2')
            all_params.append('omega3')
            all_params.append('omega22')
        if param==dust_moments_omega2_omega3:
            all_params.append('Ad')
            all_params.append('omega2')
            all_params.append('omega3')
        if param==dust_moments_omega2_omega3_bestfit:
            all_params.append('Ad')
            all_params.append('omega2')
            all_params.append('omega3')
        if param==cib_rad_MH23:
            all_params.append('Acib')

    return np.array(all_params)

#parameter labels for getdist (emcee)
def labels(param_dict):

    alllabels=[]

    for param in param_dict:

        if param==DeltaI_DeltaT:
            alllabels.append(r'dT')
        if param==DeltaI_y:
            alllabels.append(r'y')
        if param==DeltaI_reltSZ: 
            alllabels.append(r'y')
            alllabels.append(r'T_{eSZ}')
        if param==thermal_dust_rad:
            alllabels.append(r'A_d')
            alllabels.append(r'\beta_d')
            alllabels.append(r'T_d')
        if param==jens_freefree_rad:#not used
            alllabels.append(r'EM')
        if param==co_rad:#not used
            alllabels.append(r'CO')
        if param==jens_synch_rad_no_curv: #not used
            alllabels.append(r'A_s')
            alllabels.append(r'\beta_s')
        if param==cib_rad:#not used
            alllabels.append(r'A_cib')
            alllabels.append(r'\beta_cib')
            alllabels.append(r'T_cib')
        if param==dust_moments_omega2_omega3:#not used
            alllabels.append(r'A_d')
            alllabels.append(r'\omega_2')
            alllabels.append(r'\omega_3')
        if param==dust_moments_omega2_omega3_bestfit:#not used
            alllabels.append(r'A_d')
            alllabels.append(r'\omega_2')
            alllabels.append(r'\omega_3')
        if param==dust_moments_omega2_omega3_omega22:#not used
            alllabels.append(r'A_d')
            alllabels.append(r'\omega_2')
            alllabels.append(r'\omega_3')
            alllabels.append(r'\omega_{22}')
    return alllabels

#labels for plots
def legend_labels(param_dict):

    alllabels=[]

    for param in param_dict:

        if param==DeltaI_DeltaT:
            alllabels.append(r"\Delta T\,[\times 10^{-4}\,\mathrm{K}]")
        if param==DeltaI_y:
            alllabels.append(r'y\,[\times 10^{-6}]')
        if param==DeltaI_reltSZ:
            alllabels.append(r'y \,[\times 10^{-6}]')
            alllabels.append(r'T_{\rm eSZ}\,[keV]')
        if param==thermal_dust_rad:
            alllabels.append(r'A_{\rm d}\,[\times 10^{5} \mathrm{Jy/sr}]')
            alllabels.append(r'\beta_{\rm d}')
            alllabels.append(r'T_{\rm d}\,[\mathrm{K}]')
        if param==jens_freefree_rad:
            alllabels.append(r'A_{\rm FF}\,[\times 10^{5} \mathrm{Jy/sr}]')
        if param==co_rad:
            alllabels.append(r'A_{\rm CO}\,[\mathrm{Jy/sr}]')
        if param==jens_synch_rad_no_curv:
            alllabels.append(r'A_{\rm s}\,[\times 10^{5} \mathrm{Jy/sr}]')
            alllabels.append(r'\beta_{\rm s}')
        if param==cib_rad:#not used
            alllabels.append(r'A_CIB\,[\times 10^{5} \mathrm{Jy/sr}]')
            alllabels.append(r'\beta_{\rm CIB}')
            alllabels.append(r'T_{\rm CIB} [K]')
        if param==dust_moments_omega2_omega3:
            alllabels.append(r'A_{\rm d}\,[\times 10^{5} \mathrm{Jy/sr}]')
            alllabels.append(r'\omega_2')
            alllabels.append(r'\omega_3')
        if param==dust_moments_omega2_omega3_bestfit:
            alllabels.append(r'A_{\rm d}\,[\times 10^{5} \mathrm{Jy/sr}]')
            alllabels.append(r'\omega_2')
            alllabels.append(r'\omega_3')
        if param==dust_moments_omega2_omega3_omega22:
            alllabels.append(r'A_{\rm d}\,[\times 10^{5} \mathrm{Jy/sr}]')
            alllabels.append(r'\omega_2')
            alllabels.append(r'\omega_3')
            alllabels.append(r'\omega_{22}')

    return alllabels

