import numpy as np
import copy
import sys
from scipy import interpolate

FNAME=sys.argv[1]#root file name for precomputed values (this is assumed in *npy format)
N_FILES=int(sys.argv[2])#number of files to combine
N_FREQS=int(sys.argv[3])#number of frequencies
N_TEMP=int(sys.argv[4])#number of temperatures in total
FNAME_OUT=sys.argv[5]#final table file path
N_TEMP_EACH=int(N_TEMP/N_FILES) #number of temperatures in each file
print(N_TEMP_EACH)
kTe_all=[]
dI_rel=np.ones((N_FREQS,N_TEMP))

if N_FILES>1:

    for n in range(N_FILES):
    
        precompute_3D=np.load(f"{FNAME}_{n}.npy", allow_pickle=True).item()
        kTe_all.append(precompute_3D['kTe'])
        dI_rel[:,n*N_TEMP_EACH:(n+1)*N_TEMP_EACH]=np.transpose(precompute_3D['dI'])
else:
    precompute_3D=np.load(f"{FNAME}.npy", allow_pickle=True).item()
    kTe_all.append(precompute_3D['kTe'])
    dI_rel=np.transpose(precompute_3D['dI'])

rel_dict={}
rel_dict['freqs']=precompute_3D['freqs']
rel_dict['kTe']=copy.deepcopy(np.array(kTe_all).flatten())
rel_dict['dI']=copy.deepcopy(dI_rel)

np.save(f"{FNAME_OUT}.npy", rel_dict)


#now test interpolator constructed from this table
precompute_3D_tot=np.load(f"{FNAME_OUT}.npy", allow_pickle=True).item()
precompute_3D_dI=precompute_3D_tot['dI'].reshape((N_FREQS,N_TEMP))

residuals=np.zeros((N_FREQS, len(precompute_3D_tot['kTe'])))
max_residuals=np.zeros(len(precompute_3D_tot['kTe']))

for count,temp in enumerate(precompute_3D_tot['kTe']):

    rest_dI=np.delete(precompute_3D_tot['dI'],count, axis=1)
    rest_kTe=np.delete(precompute_3D_tot['kTe'], count, axis=0)
    interp_func_rest=interpolate.interp1d(rest_kTe, rest_dI, bounds_error=False, fill_value="extrapolate")

    real_dist=precompute_3D_tot['dI'][:,count]
    interp_dist=interp_func_rest(temp)

    residuals[:,count]=(interp_dist-real_dist)/real_dist*100.
    max_residuals=np.max(residuals[:,count])

print(f"maximum interpolation residuals in percent:{np.max(max_residuals)}")
