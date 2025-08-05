import jax.numpy as jnp
import constants as const
# import SZpack as SZ

# #for SZ pack
# def nu_to_x(f):
#     return const.hplanck*f/(const.kboltz*const.TCMB)

############ baseline model functions ############
############ note I_0 & jansky conversion not included, because rescaled for NUTs sampling ############ 
def dT(nu, DeltaT_amp):
    X = const.hplanck* nu/(const.kboltz*const.TCMB)
    return DeltaT_amp/const.TCMB * X**4.0 * jnp.exp(X)/(jnp.exp(X) - 1.0)**2.0 

def dust(nu, Td, Bd, Ad):
    X = const.hplanck*nu/(const.kboltz*Td)
    nu0=353.0*10.0**9
    dust = Ad*(nu/nu0)**(Bd+3.0)/(jnp.exp(X)-1.0)
    return dust

# def dust_old(nu, Td, Bd, Ad):#Abitbol+17
#     X = const.hplanck*nu/(const.kboltz*Td)   
#     dust = Ad*X**Bd*X**3.0/(jnp.exp(X)-1.0)
#     return dust

def DeltaI_y(nu, y_amp):
    X = const.hplanck*nu/(const.kboltz*const.TCMB)
    return y_amp * (X / jnp.tanh(X/2.0) - 4.0) * X**4.0 * jnp.exp(X)/(jnp.exp(X) - 1.0)**2.0   

################### additional foreground models ###################
def synch_rad(nu, As, betas):
    nu0s = 100.e9
    return As * (nu / nu0s) ** betas

def cib(nu, Tcib, Bcib, Acib):
    X = const.hplanck * nu / (const.kboltz * Tcib)
    nu0=353.0*10.0**9
    return Acib*(nu/nu0)**(Bcib+3.0)/(jnp.exp(X)-1.0)

def dust_moments_omega2_omega3(nu, Ad, Td, Bd, omega2, omega3):

    X = const.hplanck * nu / (const.kboltz * Td)
    nu0 = 353.0*10.0**9
    dIdbeta = jnp.log(nu/nu0)
    dIdT = X * jnp.exp(X) / (jnp.exp(X) - 1.)/Td
    zeroth = Ad * (nu/nu0)**(Bd+3.0) / (jnp.exp(X) - 1.)

    return zeroth * (1.+omega2*dIdbeta+omega3*dIdT)

def jens_freefree_rad(nu, EM):
    Te = 7000.
    Teff = (Te / 1.e3) ** (3. / 2)
    nuff = 255.33e9 * Teff
    gff = 1. + jnp.log(1. + (nuff / nu) ** (jnp.sqrt(3) / jnp.pi))
    return EM * gff

def co_rad(nu, Aco):
    x = jnp.load('/moto/hill/users/as6131/software/sd_foregrounds/templates/co_arrays.npy')
    freqs = x[0]
    co = x[1]
    #fs = interpolate.interp1d(jnp.log10(freqs), jnp.log10(co), bounds_error=False, fill_value="extrapolate")

    return Aco * 10. ** jnp.interp(jnp.log10(nu), jnp.log10(freqs),jnp.log10(co),left="extrapolate", right="extrapolate")

def cib_rad_MH23(nu, Acib):
    #cib sed based on best-fit results in Fiona & Colin's paper: https://arxiv.org/pdf/2307.01043.pdf
    Bcib=1.59
    Tcib=11.95
    X = const.hplanck * nu / (const.kboltz * Tcib)
    nu0=353.0*10.0**9
    return Acib * (nu/nu0)**(Bcib+3.0) / (jnp.exp(X) - 1.0)

#################### relativitistic functions (only the first one used in sampling, the other one just for testing)####################
#################### analytic one is not used in the paper in the end ####################
# def DeltaI_reltSZ(freqs, y_tot, kT_yweight, Yorder=4):
#     #based on Abitbol+2017, Hill+2015, uses Y functions of Nozawa+2006, Itoh+1998
#     yIGM_plus_yreion=1.87e-7
#     X=const.hplanck*freqs/(const.kboltz*const.TCMB)
#     Xtwid=X*jnp.cosh(0.5*X)/jnp.sinh(0.5*X)
#     Stwid=X/jnp.sinh(0.5*X)
    
#     #Y functions
#     Y0=Xtwid-4.0
#     Y1=-10.0+23.5*Xtwid-8.4*Xtwid**2+0.7*Xtwid**3+Stwid**2*(-4.2+1.4*Xtwid)
#     Y2=-7.5+127.875*Xtwid-173.6*Xtwid**2.0+65.8*Xtwid**3.0-8.8*Xtwid**4.0+11.0/30.0*Xtwid**5.0\
#     +Stwid**2.0*(-86.8+131.6*Xtwid-48.4*Xtwid**2.0+143.0/30.0*Xtwid**3.0)\
#     +Stwid**4.0*(-8.8+187.0/60.0*Xtwid)
    
#     Y3=7.5+313.125*Xtwid-1419.6*Xtwid**2.0+1425.3*Xtwid**3.0-18594.0/35.0*Xtwid**4.0+12059.0/140.0*Xtwid**5.0-128.0/21.0*Xtwid**6.0+16.0/105.0*Xtwid**7.0\
#     +Stwid**2.0*(-709.8+2850.6*Xtwid-102267.0/35.0*Xtwid**2.0+156767.0/140.0*Xtwid**3.0-1216.0/7.0*Xtwid**4.0+64.0/7.0*Xtwid**5.0)\
#     +Stwid**4.0*(-18594.0/35.0+205003.0/280.0*Xtwid-1920.0/7.0*Xtwid**2.0+1024.0/35.0*Xtwid**3.0)\
#     +Stwid**6.0*(-544.0/21.0+992.0/105.0*Xtwid)
    
#     Stwid2_term_Y4=Stwid**2.0*(-62391.0/20.0+614727.0/20.0*Xtwid-1368279.0/20.0*Xtwid**2.0+4624139.0/80.0*Xtwid**3.0-157396.0/7.0*Xtwid**4.0+30064.0/7.0*Xtwid**5.0-2717.0/7.0*Xtwid**6.0+2761.0/210.0*Xtwid**7.0)
#     Stwid4_term_Y4=Stwid**4.0*(-124389.0/10.0+6046951.0/160.0*Xtwid-248520.0/7.0*Xtwid**2.0+481024.0/35.0*Xtwid**3.0-15972.0/7.0*Xtwid**4.0+18689.0/140.0*Xtwid**5.0)
#     Stwid6_term_Y4=Stwid**6.0*(-70414.0/21.0+465992.0/105.0*Xtwid-11792.0/7.0*Xtwid**2.0+19778.0/105.0*Xtwid**3.0)
#     Stwid8_term_Y4=Stwid**8.0*(-682.0/7.0+7601.0/210.0*Xtwid)

#     Y4=-135.0/32.0+30375.0/128.0*Xtwid-62391.0/10.0*Xtwid**2.0+614727.0/40.0*Xtwid**3.0-12438.9*Xtwid**4.0+355703.0/80.0*Xtwid**5.0\
#     -16568.0/21.0*Xtwid**6.0+7516.0/105.0*Xtwid**7.0-22.0/7.0*Xtwid**8.0+11.0/210.0*Xtwid**9.0\
#     +Stwid2_term_Y4+Stwid4_term_Y4+Stwid6_term_Y4+Stwid8_term_Y4
    
#     #gfuncrel=Y0+Y1*(kT_yweight/const.m_elec)+Y2*(kT_yweight/const.m_elec)**2.0+Y3*(kT_yweight/const.m_elec)**3.0+Y4*(kT_yweight/const.m_elec)**4.0 
#     #add different y orders
#     orders=jnp.array([Y0,Y1,Y2,Y3,Y4])
#     gfuncrel=0.0
#     for i in range(Yorder+1):
#         gfuncrel+=orders[i]*(kT_yweight/const.m_elec)**i
#         print(f"added {i}")
#     if Yorder==0:
#         gfuncrel=Y0
    
#     Trelapprox = (yIGM_plus_yreion * Y0 + (y_tot-yIGM_plus_yreion) * gfuncrel) * (const.TCMB*1e6)
#     Planckian = X**4.0*jnp.exp(X)/(jnp.exp(X) - 1.0)**2.0
#     DeltaIrelapprox = Planckian*Trelapprox / (const.TCMB*1e6)

#     return DeltaIrelapprox

######################## higher order moments, not used in the analysis in the end ################################
# def dust_moments_omega2_omega3_omega33(nu, Td, Bd, Ad, omega2, omega3, omega33):

#     X = const.hplanck * nu / (const.kboltz * Td)
#     nu0 = 353.0*10.0**9
#     dIdbeta = jnp.log(nu/nu0)
#     dIdT = X * jnp.exp(X) / (jnp.exp(X) - 1.)/Td
#     dIdT_second=X*jnp.cosh(X/2.0)/jnp.sinh(X/2.0)/Td*dIdT
#     zeroth = Ad * X**Bd * X**3 / (jnp.exp(X) - 1.)
    
#     return zeroth * (1.+omega2*dIdbeta+omega3*dIdT+omega33*dIdT_second)

# def dust_moments_omega2_omega3_omega23(nu, Td, Bd, Ad, omega2, omega3, omega23):
    
#     X = const.hplanck * nu / (const.kboltz * Td)
#     nu0 = 353.0*10.0**9
#     dIdbeta = jnp.log(nu/nu0)
#     dIdT = X * jnp.exp(X) / (jnp.exp(X) - 1.)/Td
#     zeroth = Ad * X**Bd * X**3 / (jnp.exp(X) - 1.)

#     return zeroth * (1.+omega2*dIdbeta+omega3*dIdT+omega23*dIdbeta*dIdT)

# def dust_moments_omega2_omega3_omega22_omega33(nu, Td, Bd, Ad, omega2, omega3, omega22, omega33):

#     X = const.hplanck * nu / (const.kboltz * Td)
#     nu0 = 353.0*10.0**9
#     dIdbeta = jnp.log(nu/nu0)
#     dIdT = X * jnp.exp(X) / (jnp.exp(X) - 1.)/Td
#     dIdT_second=X*jnp.cosh(X/2.0)/jnp.sinh(X/2.0)/Td*dIdT
#     zeroth = Ad * X**Bd * X**3 / (jnp.exp(X) - 1.)
    
#     return zeroth * (1.+omega2*dIdbeta+omega3*dIdT+omega22*dIdbeta**2.0+omega33*dIdT_second)

# def dust_moments_omega2_omega3_omega22_omega33_omega23(nu, Td, Bd, Ad, omega2, omega3, omega22, omega33, omega23):
    
#     X = const.hplanck * nu / (const.kboltz * Td)
#     nu0 = 353.0*10.0**9
#     dIdbeta = jnp.log(nu/nu0)
#     dIdT = X * jnp.exp(X) / (jnp.exp(X) - 1.)/Td
#     dIdT_second=X*jnp.cosh(X/2.0)/jnp.sinh(X/2.0)/Td*dIdT
#     zeroth = Ad * X**Bd * X**3 / (jnp.exp(X) - 1.)

#     return zeroth * (1.+omega2*dIdbeta+omega3*dIdT+omega22*dIdbeta**2.0+omega33*dIdT_second+omega23*dIdbeta*dIdT)

# def synch_moments_omega2(nu, As, betas, omega2):
#     nu0s = 100.e9
#     zeroth=As * (nu / nu0s) ** betas
#     dIdbeta=jnp.log(nu/nu0s)
    
#     return zeroth*(1.0+omega2*dIdbeta)
