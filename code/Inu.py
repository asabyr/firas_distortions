import numpy as np
import sys
import os

file_path=os.path.dirname(os.path.abspath(__file__))
if file_path=='/moto/home/as6131/firas_distortions/code':
   fg_path=file_path.replace('/moto/home/as6131/firas_distortions/code','/moto/hill/users/as6131/software/sd_distortions')
else:
   fg_path=file_path.replace('firas_distortions/code','software/sd_distortions')
sys.path.append(fg_path)

from foregrounds import *
from spectral_distortions import *

def Inu_dust(nu, params):
    #dT+y+dust with all parameters free
    if len(params)==5:
        delta_T, y, Ad, Bd, Td=params

        dT=DeltaI_DeltaT(nu,delta_T)
        dI_y=DeltaI_y(nu,y)
        dust=thermal_dust_rad(nu, Ad, Bd, Td)

        return dT+dI_y+dust

def Inu_dT(nu, params):
    #dT only for CMB-only mocks
    dT=DeltaI_DeltaT(nu,params[0])

    return dT

def Inu_dust_SZpack(nu, params):
    
    if len(params)==6:
        delta_T, y, kTe, Ad, Bd, Td=params
        dT=DeltaI_DeltaT(nu,delta_T)
        y_rel=DeltaI_rel_SZpack(nu, y, kTe)
        dust=thermal_dust_rad(nu, Ad, Bd, Td)
    
        return dT+y_rel+dust
