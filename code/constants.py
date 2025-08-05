import numpy as np
#define some constants for numpyro and other functions
TCMB = 2.7255 #Kelvin
hplanck = 6.626070150e-34 #MKS
kboltz=1.380649e-23 # MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV!
jy = 1.e26 
dT_factor=2.0*(kboltz*TCMB)**3.0/(hplanck*clight)**2.0*jy

spectral_lines=np.array([115.27,
                230.54,
                345.8,
                424.75,
                461.04,
                492.23,
                556.89,
                576.27,
                691.47,
                809.44,
                1113.3,
                1461.1,
                1716.6,
                1900.5,
                2060.1,
                2311.7,
                2459.4,
                2589.6]) #GHz
