import sys
sys.path.append("/moto/hill/users/as6131/software/sd_foregrounds")
sys.path.append("/moto/hill/users/as6131/software/SZpack/")
import numpy as np
import constants as const
from read_data import *
import copy
import SZpack as SZ
import time
import multiprocessing
from functools import partial

#range of temperatures for which to compute
MIN_KTE=float(sys.argv[1])
MAX_KTE=float(sys.argv[2])
D_KTE=float(sys.argv[3])

#how many CORES to split across
CORES=int(sys.argv[4])

#how many NODES to split across
NODES=int(sys.argv[5])

CHUNK=int(sys.argv[6])

######################## define kTe values for which to compute #################
kTe_arr_all=np.arange(MIN_KTE, MAX_KTE, D_KTE)
if NODES>1:
    kTe_arr=np.array_split(kTe_arr_all, NODES)
else:
    kTe_arr=copy.deepcopy(kTe_arr_all)

######################## define necessary functions for distortion ########################
#sz pack needs x
def nu_to_x(f):
    return const.hplanck*f/(const.kboltz*const.TCMB)

#load firas frequencies
healpix_ost_T_lowf_invvar=prepare_data_lowf_masked_nolines("../data/monopole_firas_freq_data_healpix_orthstipes_True_20230509.pkl", 20, "invvar", [2,-1])
FREQS=copy.deepcopy(healpix_ost_T_lowf_invvar['freqs'])
x_array_FIRAS=nu_to_x(FREQS)

def compute_dI(theta, x_array=x_array_FIRAS):
#    return SZ.Integral5D.compute_from_variables(x_array, theta, 0., 0., 1.0e-6, "monopole")
    return SZ.Integral3D.compute_from_variables(x_array, theta, 0., 0., 1.0e-4, "monopole")

######################## define necessary functions for distortion ########################

def parallel_runs(thetas):
    pool = multiprocessing.Pool(processes=CORES)
    dI=partial(compute_dI)
    return pool.map(dI, thetas)

if __name__ == '__main__':

    if NODES > 1:

        dI=parallel_runs(kTe_arr[CHUNK]/const.m_elec)
        rel_dict={}
        rel_dict['freqs']=FREQS
        rel_dict['kTe']=kTe_arr[CHUNK]
        rel_dict['dI']=dI
        np.save(f"../data/3D_rel_precompute_{MIN_KTE}_{MAX_KTE}_{CHUNK}", rel_dict)

    #single node
    else:
        dI=parallel_runs(kTe_arr/const.m_elec)
        rel_dict={}
        rel_dict['freqs']=FREQS
        rel_dict['kTe']=kTe_arr
        rel_dict['dI']=dI
        np.save(f"../data/3D_rel_precompute_{MIN_KTE}_{MAX_KTE}", rel_dict)
