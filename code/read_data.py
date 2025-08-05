import numpy as np
import os
import sys
import constants as const

#define relevant paths
file_path=os.path.dirname(os.path.abspath(__file__))
fg_path=file_path.replace('/moto/home/as6131/firas_distortions/code','/moto/hill/users/as6131/software/sd_foregrounds/')
sys.path.append(fg_path)
dir_path=file_path.replace('/code','')

# some useful functions to subtract CMB spectrum 
# & convert between correlation & covariance

def CMB_BB(nu, Tcmb=2.7255):

    return 2*const.hplanck*nu**3/const.clight**2*(np.exp(const.hplanck*nu/(const.kboltz*Tcmb))-1)**(-1.)*const.jy

def corr_to_cov(corr, sigma):

    sigma=np.diag(sigma)
    dot_1=np.dot(sigma,corr)
    cov=np.dot(dot_1,sigma)

    return cov

def cov_to_corr(cov):

    D=np.diag(np.sqrt(np.diag(cov)))
    Dinv=np.linalg.inv(D)
    corr=np.dot(np.dot(Dinv, cov), Dinv)

    return corr

###############################################
# below are some functions to read monopole data, really only the functions with 
# *_masked_nolines are needed (just kept the old functions) 

#### read monopole data without any masking ####
def prepare_data_lowf(fname, sky_frac, method, BB_temp_set=True):

    data=np.load(dir_path+"/data/"+fname, allow_pickle=True)

    freqs=np.array(data['lowf']['freqs'])*1.e9
    intensity=np.array(data['lowf'][sky_frac][f"monopole_{method}"])*1.e6
    err=np.array(data['lowf'][sky_frac][f"error_{method}"])*1.e6

    if 'healpix' in fname:
        # print("healpix data")
        CMB_BB_model=CMB_BB(freqs, 2.723)
    else:
        # print("native data")
        CMB_BB_model=CMB_BB(freqs, 2.728)

    if BB_temp_set==True:
        CMB_BB_model=CMB_BB(freqs, 2.7255)

    corr=np.array(data['lowf']['freqcorr'])

    cov=corr_to_cov(corr,err)
    inv_cov=np.linalg.inv(cov)

    data_dict={}
    data_dict['cov_inv']=inv_cov
    data_dict['freqs']=freqs
    data_dict['intensity']=intensity-CMB_BB_model
    data_dict['error']=err
    data_dict['corr']=corr
    data_dict['cov']=cov

    return data_dict

def prepare_data_highf(fname, sky_frac, method, cutoff_freq, BB_temp_set=True):

    data=np.load(dir_path+"/data/"+fname, allow_pickle=True)

    freqs_all=np.array(data['high']['freqs'])*1.e9
    high_freq_mask=np.where(freqs_all<cutoff_freq*1.e9)[0]

    freqs=freqs_all[high_freq_mask]
    intensity=np.array(data['high'][sky_frac][f"monopole_{method}"])[high_freq_mask]*1.e6

    if 'healpix' in fname:
        # print("healpix data")
        CMB_BB_model=CMB_BB(freqs, 2.723)
    else:
        # print("native data")
        CMB_BB_model=CMB_BB(freqs, 2.728)

    if BB_temp_set==True:
        CMB_BB_model=CMB_BB(freqs, 2.7255)

    corr=np.array(data['high']['freqcorr'])[:len(high_freq_mask),:len(high_freq_mask)]
    err=np.array(data['high'][sky_frac][f"error_{method}"])[high_freq_mask]*1.e6

    cov=corr_to_cov(corr,err)
    inv_cov=np.linalg.inv(cov)

    data_dict={}
    data_dict['cov_inv']=inv_cov
    data_dict['freqs']=freqs
    data_dict['intensity']=intensity-CMB_BB_model
    data_dict['error']=err
    data_dict['cov']=cov

    return data_dict

######read data with masking######
def prepare_data_lowf_masked(fname, sky_frac, method, ind_mask=[0,-1]):

    data=np.load(dir_path+"/data/"+fname, allow_pickle=True)

    freqs=np.array(data['lowf']['freqs'])[ind_mask[0]:ind_mask[-1]]*1.e9
    intensity=np.array(data['lowf'][sky_frac][f"monopole_{method}"])[ind_mask[0]:ind_mask[-1]]*1.e6

    corr=np.array(data['lowf']['freqcorr'])[ind_mask[0]:ind_mask[-1],ind_mask[0]:ind_mask[-1]]
    err=np.array(data['lowf'][sky_frac][f"error_{method}"])[ind_mask[0]:ind_mask[-1]]*1.e6

    cov=corr_to_cov(corr,err)
    inv_cov=np.linalg.inv(cov)

    data_dict={}
    data_dict['cov_inv']=inv_cov
    data_dict['freqs']=freqs
    data_dict['intensity']=intensity-CMB_BB(freqs)
    data_dict['error']=err
    data_dict['corr']=corr
    data_dict['cov']=cov

    return data_dict

def prepare_data_highf_masked(fname, sky_frac, method, cutoff_freq, ind_mask):

    data=np.load(dir_path+"/data/"+fname, allow_pickle=True)

    freqs_all=np.array(data['high']['freqs'])*1.e9
    high_freq_mask=np.where(freqs_all<cutoff_freq*1.e9)[0]

    freqs=freqs_all[high_freq_mask]
    intensity=np.array(data['high'][sky_frac][f"monopole_{method}"])[high_freq_mask]*1.e6


    corr=np.array(data['high']['freqcorr'])[:len(high_freq_mask),:len(high_freq_mask)]
    err=np.array(data['high'][sky_frac][f"error_{method}"])[high_freq_mask]*1.e6
    cov=corr_to_cov(corr[ind_mask:,ind_mask:],err[ind_mask:])
    inv_cov=np.linalg.inv(cov)

    data_dict={}
    data_dict['cov_inv']=inv_cov
    data_dict['freqs']=freqs[ind_mask:]
    data_dict['intensity']=intensity[ind_mask:]-CMB_BB(freqs[ind_mask:])
    data_dict['error']=err[ind_mask:]
    data_dict['cov']=cov

    return data_dict

###########read data and remove emission lines############

#nearest index to some frequency
def find_nearest_ind(nu_all, nu_target):
    inds=[]
    for i in range(len(nu_target)):
        abs_values=np.abs(nu_all-nu_target[i])
        inds.append(np.argmin(abs_values))
    
    return np.array(inds)

#return indices contaminated by emission lines, assumes lines are in GHz, frequencies are in Hz, and 
#thresh_extra is percent value for how much a line can be broadened.

def remove_lines(freqs, thresh_extra):
    lines_Hz=const.spectral_lines*1e9
    bin_width=np.diff(freqs)[0]
    outliers=np.array([])

    for i in range(len(lines_Hz)):

        res=lines_Hz[i]-freqs

        thresh=bin_width/2+lines_Hz[i]*thresh_extra/100
        
        if np.any(np.abs(res)<thresh)==True:

            inds=np.asarray(np.abs(res)<thresh).nonzero()
            
            outliers=np.concatenate((outliers, inds[0]))

    outliers_inds=outliers.astype(int)
    return np.unique(outliers_inds)

## functions to read data, masking low & high end indices, & removing emission lines 
## fname (str) -- file name (assumes it is located in /data)
## sky_frac (float) -- sky fraction
## method (str) -- method for averaging monopole (invvar, invcov_mod, invcov_Cterm etc.)
## ind_mask (array) -- two indices, the lowest and highest frequency to include
## thresh (float) -- specificies line broadening in %
## cov_type (str) -- can be 'c', 'tot', 'calib', 'gain' so allows to test other correlation terms in covariance
## cov_file (str) -- if cov_type is something other than 'c', you need to specify cov_file to read (assuming it is in /data)
## extra_outliers_nu (float array) -- can specify specific channels to remove
## returns: data_dict (dict), which includes the following keys:
## cov_inv, freqs (GHz), intensity (Jy/sr) after subtracting CMB, error (Jy/sr),corr, cov

def prepare_data_lowf_masked_nolines(fname, sky_frac, method, ind_mask=[2,-1],
                                    thresh=1, cov_type='c', cov_file=None, extra_outliers_nu=[]):

    data=np.load(dir_path+"/data/"+fname, allow_pickle=True)

    freqs=np.array(data['lowf']['freqs'])[ind_mask[0]:ind_mask[-1]]*1.e9 #convert from GHz to Hz
    intensity=np.array(data['lowf'][sky_frac][f"monopole_{method}"])[ind_mask[0]:ind_mask[-1]]*1.e6 #convert from MJy to Jy

    orig_corr=np.array(data['lowf']['freqcorr'])[ind_mask[0]:ind_mask[-1],ind_mask[0]:ind_mask[-1]]
    err=np.array(data['lowf'][sky_frac][f"error_{method}"])[ind_mask[0]:ind_mask[-1]]*1.e6 #convert from MJy to Jy
    
    if cov_type=='c':
        print("just c cov")
        orig_corr=np.array(data['lowf']['freqcorr'])[ind_mask[0]:ind_mask[-1],ind_mask[0]:ind_mask[-1]]

    #below options were not used in the paper in the end, 
    #need to specify cov_file if you end up using these
        
    elif cov_type=='tot':
        print("tot cov")
        cov_f=np.load(dir_path+"/data/"+cov_file, allow_pickle=True)
        cov_raw=cov_f['lowf']['tot'][ind_mask[0]:ind_mask[-1],ind_mask[0]:ind_mask[-1]]
        orig_corr=cov_to_corr(cov_raw)

    else:
        print("adding c")
        cov_f=np.load(dir_path+"/data/"+cov_file, allow_pickle=True)
        cov_raw=cov_f['lowf']['c'][ind_mask[0]:ind_mask[-1],ind_mask[0]:ind_mask[-1]]

        if 'gain' in cov_type:
            cov_raw=cov_raw+cov_f['lowf']['gain'][ind_mask[0]:ind_mask[-1],ind_mask[0]:ind_mask[-1]]
            print("adding gain")
        if 'calib' in cov_type:
            cov_raw=cov_raw+cov_f['lowf']['calibrator'][ind_mask[0]:ind_mask[-1],ind_mask[0]:ind_mask[-1]]
            print("adding calib")

        orig_corr=cov_to_corr(cov_raw)

    #get freq indices contaminated by emission lines
    outliers_lowf=remove_lines(freqs,thresh)

    if len(extra_outliers_nu)>0:
        extra_outliers=find_nearest_ind(freqs, extra_outliers_nu) #any extra contamination, not used in the paper
        outliers_lowf=np.append(outliers_lowf,extra_outliers)

    freqs=np.delete(freqs, outliers_lowf)
    intensity=np.delete(intensity, outliers_lowf)
    newcorr=np.delete(orig_corr,outliers_lowf, axis=0) #remove rows
    corr=np.delete(newcorr,outliers_lowf, axis=1) #remove columns
    err=np.delete(err, outliers_lowf)

    cov=corr_to_cov(corr,err)
    inv_cov=np.linalg.inv(cov)


    data_dict={}
    data_dict['cov_inv']=inv_cov
    data_dict['freqs']=freqs
    data_dict['intensity']=intensity-CMB_BB(freqs)
    data_dict['error']=err
    data_dict['corr']=corr
    data_dict['cov']=cov

    return data_dict

## same format as previous function but ind_mask is a single integer and includes
## cutoff_freq (float) -- highest frequency to include
def prepare_data_highf_masked_nolines(fname, sky_frac, method, cutoff_freq, ind_mask=3,
                                    thresh=1, cov_type='c', cov_file=None, extra_outliers_nu=[]):

    data=np.load(dir_path+"/data/"+fname, allow_pickle=True)

    #trim based on highest frequency
    freqs_all=np.array(data['high']['freqs'])*1.e9
    high_freq_mask=np.where(freqs_all<cutoff_freq*1.e9)[0]
    freqs=freqs_all[high_freq_mask]
    intensity=np.array(data['high'][sky_frac][f"monopole_{method}"])[high_freq_mask]*1.e6
    err=np.array(data['high'][sky_frac][f"error_{method}"])[high_freq_mask]*1.e6
    
    #determine correlation matrix (orig_corr) (trimmed to highest frequency)
    if cov_type=='c':
        print("just c covariance")
        orig_corr=np.array(data['high']['freqcorr'])[:len(high_freq_mask),:len(high_freq_mask)]
    
    elif cov_type=='tot':
        print("using tot covariance")
        cov_f=np.load(dir_path+"/data/"+cov_file, allow_pickle=True)
        cov_raw=cov_f['high']['tot'][:len(high_freq_mask),:len(high_freq_mask)]
        orig_corr=cov_to_corr(cov_raw)

    else:
        print("adding c")
        cov_f=np.load(dir_path+"/data/"+cov_file, allow_pickle=True)
        cov_raw=cov_f['high']['c'][:len(high_freq_mask),:len(high_freq_mask)]

        if 'gain' in cov_type:
            print("adding gain")
            cov_raw=cov_raw+cov_f['high']['gain'][:len(high_freq_mask),:len(high_freq_mask)]
        if 'calib' in cov_type:
            print("adding calib")
            cov_raw=cov_raw+cov_f['high']['calibrator'][:len(high_freq_mask),:len(high_freq_mask)]

        orig_corr=cov_to_corr(cov_raw)

    #find channels we need to remove
    outliers_highf=remove_lines(freqs,thresh)

    if len(extra_outliers_nu)>0:
        extra_outliers=find_nearest_ind(freqs, extra_outliers_nu)
        outliers_highf=np.append(outliers_highf,extra_outliers)
        print(outliers_highf)
    
    #remove those channels from all the data
    freqs=np.delete(freqs, outliers_highf)
    intensity=np.delete(intensity, outliers_highf)
    newcorr=np.delete(orig_corr,outliers_highf, axis=0)
    corr=np.delete(newcorr,outliers_highf, axis=1)
    err=np.delete(err, outliers_highf)

    #account for masked lowest channels (this should be done at the start, 
    # but in this case emission lines are above lowest points so it works)
    cov=corr_to_cov(corr[ind_mask:,ind_mask:],err[ind_mask:])
    inv_cov=np.linalg.inv(cov)

    data_dict={}
    data_dict['cov_inv']=inv_cov
    data_dict['freqs']=freqs[ind_mask:]
    data_dict['intensity']=intensity[ind_mask:]-CMB_BB(freqs[ind_mask:])
    data_dict['error']=err[ind_mask:]
    data_dict['cov']=cov

    return data_dict
