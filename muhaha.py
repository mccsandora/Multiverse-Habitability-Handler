import os, pickle
import math, time, itertools
import numpy as np
import pandas as pd
from scipy.special import erf, erfc, expi
from scipy.integrate import quad
from tqdm import tqdm

from sample_generator import generate_random_samples
from utils import heaviside, minn, relu, ramp, rampup, signs, m3, r3, r2, cachewrap

def unpack_C(C,num):
    if C==(1,1):
        return C
    return tuple(C[:,:num])


#########           
# STARS #
#########
        
@cachewrap
def pmeas(L): # theory measure
    l,a,b,c,du,dd=tuple(L)
    return 1/b/c/du/dd
@cachewrap
def Nstars(L): # number of stars in the universe
    l,a,b,c,du,dd=tuple(L)
    return a**(-3/2)*b**(3/4)*c**3


def l_min(L): #smallest star capable of H fusion
    l,a,b,c,du,dd=tuple(L)
    return .0393*a**(3/2)*b**(-3/4)
@cachewrap
def masch(L): # imf with a knee
    l,a,b,c,du,dd=tuple(L)
    beta_imf=2.35
    return 8.9403*(1+(5.06*l_min(L)/l)**(beta_imf-1))**-1.4

# NUCLEAR EFFICIENCY
def e_nuc(L):
    l,a,b,c,du,dd=tuple(L)
    return .84-.03*a+.19*np.sqrt(.32*du+.68*dd)

# TIDAL LOCKING
def l_TL(L): # tidal locking limit
    l,a,b,c,du,dd=tuple(L)
    return .474*a**(5/2)*b**(1/2)*c**(-4/11)*e_nuc(L)**(1/11)
def l_TLb(L): # t_TL compared to t_bio
    l,a,b,c,du,dd=tuple(L)
    return .474*a**(47/17)*b**(12/17)*c**(-4/11)
@cachewrap
def f_TL(L, p_TL):
    l,a,b,c,du,dd=tuple(L)
    if p_TL==1:
        return heaviside(l-l_TL(L))
    if p_TL==2:
        return heaviside(l_TL(L)-l)
    else:
        return 1

# CONVECTION
def l_conv(L): # convective stellar mass
    l,a,b,c,du,dd=tuple(L)
    return .194*a**3*b*c**(-1/2)
@cachewrap
def f_conv(L, p_conv):
    l,a,b,c,du,dd=tuple(L)
    if p_conv==1:
        return heaviside(l-l_conv(L))
    else: 
        return 1

# BIOLOGICAL TIMESCALE
def l_bio(L): # stellar lifetime is Gyr
    l,a,b,c,du,dd=tuple(L)
    return 1.2*a**(8/5)*b**(-1/5)*c**(-4/5)*e_nuc(L)**(2/5)
@cachewrap
def f_bio(L, p_bio):  
    l,a,b,c,du,dd=tuple(L)
    if p_bio==1:
        return heaviside(l_bio(L)-l)
    else:
        return 1

# PHOTOSYNTHESIS
def l_fizzle(L,w): # star too red: w is longest wavelength in nm
    l,a,b,c,du,dd=tuple(L)
    return .21*(w/1100)**(-40/19)*a**(60/19)*b**(20/19)*c**(-10/19)
def l_fry(L,w): # star too blue: w is shortest wavelength in nm
    l,a,b,c,du,dd=tuple(L)
    return 1.75*(w/400)**(-40/19)*a**(60/19)*b**(20/19)*c**(-10/19)
@cachewrap
def f_photo(L, p_photo): 
    l,a,b,c,du,dd=tuple(L)
    if p_photo==2:
        return heaviside(l-l_fizzle(L,750))*heaviside(l_fry(L,600)-l)
    elif p_photo==1:
        return heaviside(l-l_fizzle(L,1100))*heaviside(l_fry(L,400)-l)
    else:
        return 1


##################################
# FRACTION OF STARS WITH PLANETS #
##################################
    
Z_inf=.011
def Z_min(L): # minimum metallicity for planet formation
    l,a,b,c,du,dd=tuple(L)
    return 6.3*10**-4*(1.8*l)**(3/4)*a**-3*b**(-1/2)*c**(1/2)
def f_pstep(L): # fraction of stars with planets (step function)
    return heaviside(Z_inf-Z_min(L))
def f_sn(L,Q_hat): # fraction of galaxies that can retain sn ejecta
    l,a,b,c,du,dd=tuple(L)
    return erfc(4.41*10**(-5)*a**2*b**(5/3)/Q_hat)
@cachewrap
def f_p(L,Q_hat): # fraction of stars with planets
    return f_sn(L,Q_hat)*f_pstep(L)

# HOT JUPITERS
def Z_maxpp(L,kappa_hat): # maximum metallicity for no hot jupiters (planet-planet interactions)
    l,a,b,c,du,dd=tuple(L)
    return .12/(1.8*l)*a**(13/6)*b**(-3/2)/kappa_hat
@cachewrap
def f_hj(L, kappa_hat, p_hj): # simplified fraction of earths without hot jupiters
    if p_hj==1:
        return (1-(Z_inf/Z_maxpp(L,kappa_hat))**2)*heaviside(Z_maxpp(L,kappa_hat)-Z_inf)
    else:
        return 1


###############################
# NUMBER OF HABITABLE PLANETS #
###############################

# ISOLATION MASS
qiso=1/3 # slope of planetesimal masses
p_sigma=0 # planet variance set by 0: entropy or 1: shot noise
p_ldep=1 # include stellar mass-planet mass correlation 0: no 1: yes
def r_iso(L,kappa_hat): # ratio of isolation mass to terrrestrial mass
    l,a,b,c,du,dd=tuple(L)
    return .1*(1.8*l)**2*a**-4*b**(-21/8)*kappa_hat**(3/2)
def r_inn(L,kappa_hat): # ratio of innder disk mass to isolation mass
    l,a,b,c,du,dd=tuple(L)
    return 30*(1.8*l)**(-1/3)*a**(5/6)*b**(5/8)*kappa_hat**(-1/2)
def fiso(x): # fraction of terrestrial planets
    return min(1,(x/.3)**qiso)-min(1,(x/4)**qiso)
f_iso=np.vectorize(fiso)
def n_iso(x): # average number of planets
    return (qiso-1)/qiso*(x-x**(1-qiso))/(1-x**(1-qiso))

# GIANT IMPACT MASS
def r_olig(L,kappa_hat): # ratio of oligarch mass to terrestrial mass
    l,a,b,c,du,dd=tuple(L)
    return 2.64*(1.8*l)**(p_ldep*5/2)*a**(-9/2)*b**(-45/16)*kappa_hat**(3/2) # accretion
#    return 2.64*(1.8*l)**(p_ldep*139/40)*a**(-39/4)*b**(-3)*c**(3/4) #irradiation
def f_olig(x,s): # number of terrestrial mass oligarchs
    return 1/2*(erf((np.log(4/x)+s**2/2)/np.sqrt(2)/s)\
                -erf((np.log(.3/x)+s**2/2)/np.sqrt(2)/s))
#    return np.exp(-np.pi/4*(.3/x)**2)-np.exp(-np.pi/4*(4/x)**2) # rayleigh dist
def sigma_m(L, kappa_hat): #variance of m_olig
    return [1/np.sqrt(6),1/np.sqrt(r_inn(L, kappa_hat))]
def n_olig(L, kappa_hat): # average number of planets 
    l,a,b,c,du,dd=tuple(L)
    return 3*(1.8*l)**(-5/6)*a**(4/3)*b**(13/16)*kappa_hat**(-1/2) # accretion
#    return 3*(1.8*l)**(-11/8)*a**(17/4)*b**(11/12)*c**(-5/12) # irradiation

@cachewrap
def n_terr(L, kappa_hat, p_terr):
    if p_terr==2:
        return n_iso(r_inn(L, kappa_hat))*f_iso(r_iso(L,kappa_hat))
    elif p_terr==1:
        return n_olig(L, kappa_hat)*f_olig(r_olig(L, kappa_hat),sigma_m(L, kappa_hat)[p_sigma])
    else:
        return 1

# TEMPERATE ZONE
def temp_thresh(L, kappa_hat): # temperate zone smaller than disk size
    l,a,b,c,du,dd=tuple(L)
    return .01*(1.8*l)**(17/12)*a**(-5)*b**(-2)*c**(1/2)*kappa_hat
@cachewrap
def n_temp(L, kappa_hat, p_temp): # number of planets in temperate zone
    l,a,b,c,du,dd=tuple(L)
    if p_temp==1:
        return .431*l**(-85/48)*a**(11/2)*b**(7/4)*c**(-5/8)*kappa_hat**(-1/2)\
    *heaviside(1-temp_thresh(L, kappa_hat))
    else:
        return 1


########
# LIFE #
########
    
@cachewrap
def r_time(L, p_time): # ratio of stellar to molecular timescale
    l,a,b,c,du,dd=tuple(L)
    if p_time==1:
        return l**(-5/2)*a**(4)*b**(-1/2)*c**-2*e_nuc(L)
    else:
        return 1
@cachewrap
def r_area(L, p_area): # ratio of planet area to molecular area
    l,a,b,c,du,dd=tuple(L)
    if p_area==1:
        return a**(3/2)*b**(3/4)*c**-3 # L_mol~a_0
    else:
        return 1
@cachewrap
def S_tot(L, p_S, ec=1): # entropy produced in stellar lifetime
    l,a,b,c,du,dd=tuple(L)
    S = (1.8*l)**(-119/40)*a**(17/2)*b**2*c**(-17/4)*e_nuc(L)
    C = (1.8*l)**(-5/2)*a**(9/2)*b**(1/2)*c**(-3)*e_nuc(L)
    if p_S==1:
        return S
    elif p_S==2:
        return C
    elif p_S==3:
        return minn(S,ec*C)
    else:
        return 1
def C_tot(L): # material limited biosphere
    l,a,b,c,du,dd=tuple(L)
    return (1.8*l)**(-5/2)*a**(9/2)*b**(1/2)*c**(-3)*e_nuc(L)
def f_plates(L, p_plates):
    l,a,b,c,du,dd=tuple(L)
    if p_plates==1:
        return 1
        # return heaviside(a_ptmax-a)*heaviside(a-a_ptmin)
    else:
        return 1

# OXYGENATION TIME
r_o=.22 # Earth's ratio
def rip(x): # regulates the exp in rat_t
    return 500*np.tanh(x/500)
def rat_tup(L): # ratio of oxidation time to stellar lifetime
    l,a,b,c,du,dd=tuple(L)
    return r_o*(1.8*l)**(81/40)*a**(-4)*b**(3/4)*c**(3/4)*\
    np.exp(rip(-18.85*(b**(-1/2)-1)\
               +15.37*((1.8*l)**(-19/40)*a**(3/2)*b*c**(-1/4)-1)))/e_nuc(L)
def rat_tdown(L): # ratio of timescales with drawdown
    l,a,b,c,du,dd=tuple(L)
    return r_o*(1.8*l)**(5/2)*a**-3*b**(1/4)\
               *np.exp(18.85*(1-b**(-1/2)))/e_nuc(L)
#def rat_t(L, p_O2):
#    l,a,b,c,du,dd=tuple(L)
#    return [0,rat_tdown(L),rat_tup(L),\
#            2/(1/(rat_tdown(L)+10**-6)+1/rat_tup(L))]
@cachewrap
def f_O2(L, p_O2):
    l,a,b,c,du,dd=tuple(L)
    if p_O2==1:
        return relu(1-rat_tdown(L))
    elif p_O2==2:
        return relu(1-rat_tup(L))
    elif p_O2==3:
        return relu(1-2/(1/(rat_tdown(L)+10**-6)+1/rat_tup(L)))
    else:
        return 1

#########
# DEATH #
#########
    
Myr=1
km=1
TW=1
def t_rec(L): # recovery time
    l,a,b,c,du,dd=tuple(L)
#    return 10*a**(-2)*b**(-3/2)*Myr #mol
    return 10*(1.8*l)**(17/8)*a**(-15/2)*b**(-3)*c**(-1/4)*Myr # yr
def t_star(L): # stellar lifetime
    l,a,b,c,du,dd=tuple(L)
    return 5500*(1.8*l)**(-5/2)*a**2*b**(-2)

# COMETS
def d_comet(L, kappa_hat): # typical comet size
    l,a,b,c,du,dd=tuple(L)
    return 1*(1.8*l)*a**(-25/9)*b**(-3/2)*c**(-5/6)*kappa_hat**(2/3)*km
def d_sox(L): # extinction size (sulfates)
    l,a,b,c,du,dd=tuple(L)
    return 1*(1.8*l)**(1/4)*a**(-3)*b**(-1)*c**(-1/2)*km
def d_dust(L): # extinction size (dust)
    l,a,b,c,du,dd=tuple(L)
    return 1*(1.8*l)**(1/4)*a**(-5/3)*b**(-1)*c**(-1/2)/\
(1/2+1/2*erfc(.57*(.769+np.log(.01*a**(1/2)*b**(1/4)*c**(-1)))))**(1/3)*km
def d_both(L):
    l,a,b,c,du,dd=tuple(L)
    return minn(d_sox(L),d_dust(L))
def G_comets(L, kappa_hat): # comet extinction rate
    l,a,b,c,du,dd=tuple(L)
    return (1.8*l)**(-16/5)*a**8*b**(-1/2)*kappa_hat**(3/2)*(ramp(d_comet(L, kappa_hat)/d_both(L)))**(1.5)/(3*Myr)/30

# GRBS
def vh(x): # volume function
    if x<=.075:
        return 40/3*x**3
    if .075<x<1:
        return x**2
    if 1<=x:
        return 1
vg=np.vectorize(vh)
def rat_grb(L, Q_hat, kappa_hat): # ratio of grb death radius to galaxy radius 
    l,a,b,c,du,dd=tuple(L)
    return 1/6*a**(-1)*b**(-3/2)*c**(-1/2)*kappa_hat**(3/2)*Q_hat**(-1/2)
def G_grb(L, Q_hat, kappa_hat): # grb extinction rate
    l,a,b,c,du,dd=tuple(L)
    return c*Q_hat**(3/2)*36*vg(rat_grb(L, Q_hat, kappa_hat))/(90*Myr)

# GLACIATIONS
def Q_dirty(L): # dimensional analysis estimate of heat flux 
    l,a,b,c,du,dd=tuple(L)
    return 47*a**(9/2)*b**(7/2)*c**(-1)*TW
def s_t(L): # dimensionless quantity in heat flux
    l,a,b,c,du,dd=tuple(L)
    return .4242*a**(1/2)*b**(5/8)*c*(t_rec(L)/Myr)**(1/2)
def Q_form(L): # heat of formation
    l,a,b,c,du,dd=tuple(L)
    return Q_dirty(L)*5.13/s_t(L)*math.e**(-s_t(L))
def f_rad(L): # dimensionless fraction in radioactive heat
    l,a,b,c,du,dd=tuple(L)
    return math.e**(1+52.72*(1-(137/144/a)**(1/2))\
                    -math.e**(52.72*(1-(137/144/a)**(1/2))))
def Q_rad(L): # radioactive heat
    l,a,b,c,du,dd=tuple(L)
    return 47*a**(3/2)*b**(3/4)*c**(-3)*f_rad(L)*TW
def Q_both(L):
    l,a,b,c,du,dd=tuple(L)
    return (Q_form(L)+Q_rad(L))/2
def G_glac(L): # glaciation extinction rate
    l,a,b,c,du,dd=tuple(L)
    return Q_both(L)/(47*TW)*a**(-7/2)*b**(-9/4)*c**3/(90*Myr)

# VOLCANOES
def G_vol(L): # volcano extinction rate
    l,a,b,c,du,dd=tuple(L)
    return (Q_both(L)/(47*TW))**(1/2)*a**(-3/4)*b**(-3/8)*c**(3/2)/(90*Myr)

# TOTAL
def G_death(L, Q_hat, kappa_hat, p_comets, p_grb, p_glac, p_vol): # total extinction rate
    l,a,b,c,du,dd=tuple(L)
    return (p_comets*G_comets(L, kappa_hat)+p_grb*G_grb(L, Q_hat, kappa_hat)\
            +p_glac*G_glac(L)+p_vol*G_vol(L))/(p_comets+p_grb+p_glac+p_vol+10**-6)
def f_setback(t,G):
    return (1-t*G)*heaviside(1-t*G)
def f_IDH(t,G):
    return (t*G+10**-6)*(1-t*G)*heaviside(1-t*G)
def f_reset(t,ts,G):
    return 1-(1-np.exp(-10*t*G))**(ts/(10*t))
@cachewrap
def f_int(L, Q_hat, kappa_hat, p_death, p_comets, p_grb, p_glac, p_vol): # fraction of life that develops intelligence
    l,a,b,c,du,dd=tuple(L)
    if p_death==1:
        return f_setback(t_rec(L),G_death(L, Q_hat, kappa_hat, p_comets, p_grb, p_glac, p_vol))
    elif p_death==2:
        return f_IDH(t_rec(L),G_death(L, Q_hat, kappa_hat, p_comets, p_grb, p_glac, p_vol))
    elif p_death==3:
        return f_reset(t_rec(L),t_star(L),\
                       G_death(L, Q_hat, kappa_hat, p_comets, p_grb, p_glac, p_vol))
    else:
        return 1

############
# ELEMENTS #
############
        
r_rm=1.9
if 'gamma_inv' not in globals():
    gamma_inv = np.array([[quad(lambda x: np.exp(x)*x**(.54-1),0,i)[0],i] 
                     for i in np.arange(.01,20,.001)])

def Delta_ER(L):
    l,a,b,c,du,dd=tuple(L)
    return 1.35*du+2.92*dd+3.97*a-8.24

def P31_stable(L):
    l,a,b,c,du,dd=tuple(L)
    return 8.25589*a+0.511*b-4.67*dd+2.16*du

def S32_stable(L):
    l,a,b,c,du,dd=tuple(L)
    return -5.025+7.95682*a-0.511*b-4.67*dd+2.16*du

def P32_stable(L):
    l,a,b,c,du,dd=tuple(L)
    return -(-6.575+7.50703*a-0.511*b-4.67*dd+2.16*du)

def N14_stable(L):
    l,a,b,c,du,dd=tuple(L)
    return -1.58793+4.76494*a-0.511*b-4.67*dd+2.16*du

def Cl35_stable(L):
    l,a,b,c,du,dd=tuple(L)
    return -(-5.30286+8.19482*a-0.511*b-4.67*dd+2.16*du)
    
def Fe_stable(L): 
    l,a,b,c,du,dd=tuple(L)
    return -(-9.5+10.5412*a-0.511*b-4.67*dd+2.16*du)

def Co_unstable(L): 
    l,a,b,c,du,dd=tuple(L)
    return -3.75714+10.9144*a-0.511*b-4.67*dd+2.16*du

def Ni_unstable(L): 
    l,a,b,c,du,dd=tuple(L)
    return -2.87143+11.2876*a-0.511*b-4.67*dd+2.16*du

def not_Cr(L):
    l,a,b,c,du,dd=tuple(L)
    return -(-10.3857+10.1679*a-0.511*b-4.67*dd+2.16*du)

    # old
    #Delta_ER = 1.35*du+2.92*dd+3.97*a-7.86
    #N14_stable = -1.77143+4.76494*a-0.511*b-4.67*dd+2.16*du
    #Cl36_stable1 = (-4.46667+8.56201*a-0.511*b-4.67*dd+2.16*du)
    #Cl36_stable2 = -(-5.84444+8.12954*a-0.511*b-4.67*dd+2.16*du)

def gi(x):
    a  = np.searchsorted(gamma_inv[:,0],x)
    return gamma_inv[a,1]
@cachewrap
def f_mr(L, kappa_hat, p_metal): #metal to rock ratio
    l,a,b,c,du,dd=tuple(L)
    if p_metal==1:
        return 1-np.exp(-gi(.638*r_rm*a**(.56)*b**(-.82)*c**(-.54)*kappa_hat**.81))
    else:
        return 1

@cachewrap
def f_CO(L,p_CO,k_C=-.0089,k_O=.0037,k_Mg=-.500,k_Si=.0062):
    if p_CO==1:
        return heaviside(Delta_ER(L)-k_C)*heaviside(k_O-Delta_ER(L))
    elif p_CO==2:
        return heaviside(Delta_ER(L)-k_Mg)*heaviside(k_Si-Delta_ER(L))
    else:
        return 1
    
@cachewrap
def f_NPS(L,p_NPS):    
    if p_NPS==1:
        return heaviside(N14_stable(L))
        #return heaviside(Cl36_stable1)*heaviside(Cl36_stable2)
    if p_NPS==2:
        return heaviside(P31_stable(L))+\
               51*heaviside(-S32_stable(L))*heaviside(P32_stable(L))+\
                .4*heaviside(-P32_stable(L))
    elif p_NPS==3: 
        return heaviside(S32_stable(L))+\
               heaviside(-P31_stable(L))+heaviside(Cl35_stable(L))
    else:
        return 1

@cachewrap
def f_Fe(L,p_Fe):
    if p_Fe==1:
#        return heaviside(Ni_unstable(L))*heaviside(Fe_stable(L))
        return heaviside(Co_unstable(L))*heaviside(Fe_stable(L))
#        return heaviside(not_Cr(L))
#        return heaviside(Co_unstable(L))
    else:
        return 1
        
##########################
# PLANETARY HABITABILITY #
##########################
        
@cachewrap
def f_ecc(L, kappa_hat, p_ecc,e_max=.25):
    l,a,b,c,du,dd=tuple(L)
    e_bar = .04*kappa_hat**(1/2)*(1.8*l)**(17/16)*a**(-7/2)*b**(-5/4)*c**(3/8)
    if p_ecc==1:
        return 1-(1+2*e_max/e_bar)*np.exp(-2*e_max/e_bar)
        #return 1-np.exp(-(e_max/e_bar)**2)
    else:
        return 1

def om_earth_jup_rat(L, kappa_hat):
    l,a,b,c,du,dd=tuple(L)
    return .62*kappa_hat**(-1/3)*(1.8*l)**(-3.44)*a**(11.03)*b**(2.76)*c**(-1.48)
    
def om_moon_jup_rat(L, kappa_hat):
    l,a,b,c,du,dd=tuple(L)
    return 1.34*kappa_hat**(7/6)*(1.8*l)**(3.93)*a**(-9.97)*b**(-3.99)*c**(.77)

def f_moon(L, kappa_hat):
    l,a,b,c,du,dd=tuple(L)
    v_rat = 9.33*kappa_hat**(-1/2)*(1.8*l)**(-11/16)*a**2*b*c**(-1/8)
    t_rat = 12.4*kappa_hat**(-3/2)*(1.8*l)**(103/12)*a**(-41/2)*b**(-9/4)*c**(7/2)
    return .014*heaviside(v_rat-1)*heaviside(t_rat-1)

@cachewrap
def f_obliquity(L, kappa_hat, p_obliquity):
    if p_obliquity==1:
        return heaviside(om_earth_jup_rat(L, kappa_hat)+om_moon_jup_rat(L, kappa_hat)-1)*\
               (f_moon(L, kappa_hat)*heaviside(1-om_earth_jup_rat(L, kappa_hat))+\
               heaviside(om_earth_jup_rat(L, kappa_hat)-1))
        #return f_moon(L) 
    else:
        return 1

def Hw(f_h2o,k_h2o):
    fw = 1 - np.exp(-9.52*f_h2o)
    hw = [fw*(1-fw),
          fw**6*(1-fw)**3,
          (1-fw)**2,
          fw**2,
          heaviside(.92-fw),
          heaviside(fw-.05)]
    return hw[k_h2o]

def Fneb(q):
    return 2/3-1/6*np.exp(1/q-1)*q*(1+q+2*q**2)+1/(6*math.e)*(expi(1/q)-expi(1))

@cachewrap
def f_ocean(L,kappa_hat, p_ocean,r_damp,k_h2o):
    l,a,b,c,du,dd=tuple(L)
    if p_ocean==1:
        z_damp = kappa_hat**(3/2)*(1.8*l)**(-167/40)*a**(21/2)*b**(-13/4)*c**(-5/4)
        f_damp = rampup(1-(r_damp*z_damp)**(.4))/(1-(r_damp)**(.4))
        f_h2o = .13*kappa_hat*(1.8*l)**(21/10)*a**(-11/2)*b**(-25/12)*c**(1/3)*f_damp
        return Hw(f_h2o,k_h2o)
    if p_ocean==2:
        z_damp = kappa_hat**(3/2)*(1.8*l)**(-167/40)*a**(21/2)*b**(-13/4)*c**(-5/4)
        f_damp = rampup(1-(r_damp*z_damp)**(.4))/(1-(r_damp)**(.4))
        f_h2o = .13*kappa_hat*(1.8*l)**(21/10)*a**(-11/2)*b**(-25/12)*c**(1/3)*f_damp
        
        t_mig_rat = .89*kappa_hat**(5/2)*(1.8*l)**(29/20)*a**(-4)*b**(-23/6)*c**(-1/6)
        f_grand_tack = .5*(np.exp(-.24*t_mig_rat)-np.exp(-1.76*t_mig_rat))
        return Hw(f_h2o,k_h2o)*f_grand_tack
    if p_ocean==3:
        f_h2o = .13*(1.8*l)**(-31/20)*a**(9/2)*b**(-1/4)*c**(-1/2)
        return Hw(f_h2o,k_h2o)
    if p_ocean==4:
        f_h2o = .13*(1.8*l)**(-1141/480)*a**(47/8)*b**(-1/2)*c**(-13/16)*\
          Fneb(10/rip(10/(1/16*(1.8*l)**(-19/80)*a**(3/4)*b**(1/2)*c**(-1/8))))/69.76
        return Hw(f_h2o,k_h2o)
    else:
        return 1
    
############################
# ATMOSPHERIC HABITABILITY #
############################

        
''' ALL QUANTITIES HERE ARE DIVIDED BY 
    M_EARTH, AND EXPRESSED IN TERMS OF M_ATM.'''

def f_spot(L):
    l,a,b,c,du,dd=tuple(L)
    roe = np.exp(rip(3.84074*(1.0*l)**(19/40)*a**(-3/2)*b**(-1)*c**(1/4)))
    fs = 0.5344*np.exp(-0.236147*roe*(1.0*l)**(11/80)*a**(-3/4)*b**(1/4)*c**(1/8))
    fs[fs<10**-60]=10**-60
    return fs

        
@cachewrap
def f_atm(L, kappa_hat, p_atm,k_atm_source,k_atm_N,k_atm_min):
    l,a,b,c,du,dd=tuple(L)
    if k_atm_source==0:
        m_atm = kappa_hat*(1.8*l)**(21/10)*a**(-11/2)*b**(-25/12)*c**(1/3)*f_NPS(L,1)**k_atm_N
    elif k_atm_source==1:
        m_atm = kappa_hat*(1.8*l)**(25/12)*a**(-9)*b**(-2)*c**(1/2)*f_NPS(L,1)**k_atm_N
    m_triple = .006*b**(-1/2)*np.exp(18.15*(1-b**(-1/2)))
    m_diurnal = .005*a**2*b**(3/4)
#    m_tropo = .128*a**(-9/2)*b**(-1/4)*c
#    m_xray = .038*(1.8*l)**(-181/80)*a**(23/4)*b**(1/2)*c**(-7/8)
#    m_xrayb = .038*(1.8*l)**(19/80)*a**(7/4)*b*c**(9/8)
    m_xray0 = .038*(1.8*l)**(-71/40)*a**(27/4)*b**(1/4)*c**(-1)*(f_spot(L)/.01)**(3/2)
    m_xrayconv = .038*2.01*(1.8*l)**(-131/80)*a**(6)*b**(1/2)*c**(-7/8)\
                 *np.exp(rip(3.84*(l**(19/40)*a**(-3/2)*b**(-1)*c**(1/4)-(1/1.8)**(19/40))))*(f_spot(L)/.01)**(3/2)
    m_xray = minn(m_xray0,m_xrayconv)
    if k_atm_min==0:
        m_min = m_xray
    if k_atm_min==1:
        m_min = m_triple
    if k_atm_min==2:
        m_min = m_diurnal
#    m_max = maxx(m_triple,m_diurnal,m_xray)
    if p_atm==1:
        return heaviside(m_atm-m_min)
    if p_atm==2:
        m_atm *= .0095
        return ramp(m_atm/(m_min+.0000001))        
        #return heaviside(m_atm-m_min)+m_atm/(m_min+.0000001)*heaviside(m_min-m_atm)
        #return m_atm2/(m_xray+.0000001)
    else:
        return 1
    
@cachewrap
def f_B(L, kappa_hat, p_B):
    l,a,b,c,du,dd=tuple(L)
    r_mp2pl = 10*(1.8*l)**(23/60)*a**(-13/12)*b**(-11/24)*c**(1/6)
    Ra = 1000*a**(7/3)*b**(7/6)*c**(-2/3)
    r_temp2alf = 8.96*(1.8*l)**(3/8)*a**(-11/4)*b**(-2)*c**(1/2)*(f_spot(L)/.01)**(-1/2)
    R2 = .163*kappa_hat**-.81*a**-.56*b**.82*c**.54
    Delta_Emax = 27.33*R2-.874
    core_exists = 1-heaviside(Delta_ER(L)-Delta_Emax)*heaviside(.62-R2)
    r_tsolid2tstar = 2*(1.8*l)**(5/2)*a**(-3)*b**(3/4)
    if p_B==1:
        return heaviside(Ra-100)*heaviside(r_mp2pl-1)*heaviside(r_temp2alf-1)\
               *core_exists*heaviside(r_tsolid2tstar-1)
    else:
        return 1
    

##################
# ORIGIN OF LIFE #
##################

@cachewrap
def f_ool(L, kappa_hat, p_ool,k_SEP):
    l,a,b,c,du,dd=tuple(L)
    if p_ool==1: # lightning
        return (1.8*l)**(-5/2)*a**(7)*b**(3/2)*c**(-4)
    if p_ool==2: # sep
        cE = (1.8*l)**(-17/20)*a**(5/2)*b**(-1/12)*c**(1/3)
        return (1.8*l)**(-69/20)*a**(19/2)*b**(3/2)*c**(-4)*(cE**k_SEP)/f_spot(L)
        #return (1.8*l)**(-43/10)*a**(12)*b**(17/12)*c**(-11/3)/f_spot(L)    
    if p_ool==3: # xuv
        return (1.8*l)**(-23/20)*a**(17/2)*b**(3/2)*c**(-4)/f_spot(L)
    if p_ool==4: # vents
        return (1.8*l)**(-5/2)*a**(9/2)*c**(-3)*heaviside(Delta_ER(L)+.874)*f_Fe(L,1)*f_NPS(L,3)
    if p_ool==5: # idp
        sigma_chem = 660*(1.8*l)**(-79/96)*a**(35/8)*b**(31/24)*c**(-53/48)
        return (1.8*l)**(-31/20)*a**(6)*b**(1/2)*c**(-7/2)*heaviside(sigma_chem-75)
    if p_ool==6: # comets
        return (1.8*l)**(-23/10)*a**(9)*b**(1)*c**(-4)
    if p_ool==7: # asteroids
        return kappa_hat*(1.8*l)**(21/10)*a**(-4)*b**(-4/3)*c**(-8/3)
    if p_ool==8: # moneta
        return kappa_hat**(4/3)*(1.8*l)**(619/60)*a**(-34)*b**(-55/4)*c**(-3/4)
    if p_ool==9: # interplanetary panspermia
        return kappa_hat**(5/8)*(1.8*l)**(-499/480)*a**(257/24)*b**(-15/16)*c**(-35/16)        
    if p_ool==10: # interstellar panspermia
        return kappa_hat**(9/8)*(1.8*l)**(-33/160)*a**(361/24)*b**(-7/4)*c**(-51/16)        
    else:
        return 1

####################
# EXOTIC OBSERVERS #
####################

@cachewrap
def f_binary(L,p_bin):
    l,a,b,c,du,dd=tuple(L)
    if p_bin==1:
        return 1/(1+1*a**(-25/21)*b**(29/42)*c**(2/21))
    else:
        return 1
    
@cachewrap
def f_water(L, kappa_hat, p_water):
    l,a,b,c,du,dd=tuple(L)
    e_h2o = a**2*b**(3/2)

    if p_water==1:
        return heaviside(1.11-e_h2o)*heaviside(e_h2o-.89) # K/Na/Cl pumps
    if p_water==2:
        return heaviside(e_h2o-.98) # ice floats
    if p_water==3:
        return heaviside(1.03-e_h2o) # viscosity increased 23%
    if p_water==4:
        return heaviside(e_h2o-.98)*heaviside(1.03-e_h2o) #viscosity + ice

    if p_water==5: # al26
        t_al26 = 1+7.29*(b-1)+26.03*(1.92*dd-.92*du-1)\
    -25.72*0-75.32*(a-1)
        return .014*heaviside(t_al26-.1)*heaviside(10-t_al26)
    if p_water==6: # liquid surface water body fraction
        ww = 10**-4*kappa_hat*(1.8*l)**(17/12)*a**(-5)*b**(-2)*c**(1/2)
        return ww
    else:
        return 1
    
@cachewrap
def f_rogue(L, kappa_hat, p_rogue):
    l,a,b,c,du,dd=tuple(L)
    if p_rogue==1:
        x = a*b/kappa_hat
        return 10*x**(3/10)*heaviside(1-.018*x)
    else:
        return 1
    
@cachewrap
def f_icy_moons(L, kappa_hat, p_icymoons):
    l,a,b,c,du,dd=tuple(L)
    if p_icymoons==1:
        r_ice_disk = .027*kappa_hat*(1.8*l)**(11/10)*a**(-4)*b**(-4/3)*c**(1/3)
        r_ice_moon = 20/250*kappa_hat**(-1/3)*(1.8*l)**.24*a**-2.11*b**.62*c**1.41
        #N_ice_moon = 6*kappa_hat**(-1/3)*(1.8*l)**.91*a**-1.95*b**-.01*c**.23
        N_ice_moon = 6*kappa_hat**(-1/3)*(1.8*l)**1.18*a**-2.70*b**-.09*c**.31
        return N_ice_moon*\
    (1-r_ice_disk)/(1-.027)*\
    heaviside(1-r_ice_disk)*\
    heaviside(5*20/250-r_ice_moon)
    else:
        return 1
    
@cachewrap
def f_stellar_surface(L,p_ss):
    l,a,b,c,du,dd=tuple(L)
    ss = 21.17*(1.8*l)**(19/40)*a**(-3/2)*b**(-1)*c**(1/4)
    if p_ss==1:
        return heaviside(1-ss)
    else:
        return 1
    
#######################
# NUMBER OF OBSERVERS #
#######################

def probs_nobs(LB, P_list, CB=(1,1), Q_l=1, Q_ER=0, n_hard=1,
               k_C=-.0089, k_O=.0037, k_Mg=-.500, k_Si=.0062,
               e_max=.35, r_damp=.5, k_h2o=0,
               k_atm_source=0, k_atm_N=0, k_atm_min=0,
               k_SEP=1,
               k_guest=0,
               p_measure=pmeas,
               return_ps=True):
    p_photo,p_TL,p_conv,p_bio,p_plates,\
    p_hj,p_terr,p_temp,p_time,p_area,p_S,\
    p_O2,p_death,p_comets,p_grb,p_glac,p_vol,\
    p_metal,p_CO,p_NPS,p_Fe,\
    p_eccentricity,p_obliquity,p_ocean,\
    p_atm,p_B,\
    p_ool,\
    p_water,p_binary,p_icy_moons,p_rogue,\
    p_guest = P_list

    L = LB[p_plates+2*(p_CO>0)]
    l,a,b,c,du,dd=tuple(L)
    Q_hat, kappa_hat = unpack_C(CB,len(l))

    n = p_measure(L)\
        *Nstars(L)\
        *masch(L)\
        *f_photo(L, p_photo)\
        *f_TL(L, p_TL)\
        *f_conv(L, p_conv)\
        *f_bio(L, p_bio)\
        *f_plates(L, p_plates)\
        *f_p(L, Q_hat)\
        *f_hj(L, kappa_hat, p_hj)\
        *n_terr(L, kappa_hat, p_terr)\
        *n_temp(L, kappa_hat, p_temp)\
        *r_time(L, p_time)**n_hard\
        *r_area(L, p_area)**n_hard\
        *S_tot(L, p_S)**n_hard\
        *f_O2(L, p_O2)\
        *f_int(L, Q_hat, kappa_hat, p_death, p_comets, p_grb, p_glac, p_vol)\
        *f_mr(L, kappa_hat, p_metal)\
        *f_CO(L, p_CO, k_C, k_O, k_Mg, k_Si)\
        *f_NPS(L, p_NPS)\
        *f_Fe(L, p_Fe)\
        *f_ecc(L, kappa_hat, p_eccentricity, e_max)\
        *f_obliquity(L, kappa_hat, p_obliquity)\
        *f_ocean(L, kappa_hat, p_ocean, r_damp, k_h2o)\
        *f_atm(L, kappa_hat, p_atm, k_atm_source, k_atm_N, k_atm_min)\
        *f_B(L, kappa_hat, p_B)\
        *f_ool(L, kappa_hat, p_ool,k_SEP)\
        *f_water(L, kappa_hat, p_water)\
        *f_binary(L, p_binary)\
        *f_rogue(L, kappa_hat, p_rogue)\
        *f_icy_moons(L, kappa_hat, p_icy_moons)
    
    if return_ps:
        D = np.sum(n)
        pa = np.sum(n[a>=1])/D
        pb = np.sum(n[b>=1])/D
        pc = np.sum(n[c>=1])/D
        pdu = np.sum(n[du>=1])/D
        pdd = np.sum(n[dd>=1])/D
        pall = [m3(pa),m3(pb),m3(pc),m3(pdu),m3(pdd)]
        if Q_l==1:
            pl = probs_l(L, P_list, CB, n_hard, 
                         k_C,k_O,k_Mg,k_Si, 
                         k_h2o, 
                         k_atm_source, k_atm_N, k_atm_min,
                         k_SEP,
                         k_guest,
                         p_measure)
            pall.append(pl)
        if Q_ER==1:
            Delta_ER = 1.35*du+2.92*dd+3.97*a-8.24
            pER = np.sum(n[Delta_ER>0])/D
            pER2 = np.sum(n[(Delta_ER>0) & (Delta_ER<.0035)])/D
            pall.append(m3(pER))     
            pall.append(m3(pER2))
        return pall
    else:
        return n

def probs_l(L, P_list, CB, n_hard, 
            k_C,k_O,k_Mg,k_Si, 
            k_h2o, 
            k_atm_source, k_atm_N, k_atm_min,
            k_SEP,
            k_guest,
            p_measure): 
    # probability of being around a sunlike star within our universe
    l,a,b,c,du,dd=tuple(L)    
    lo = l_min([1,1,1,1,1,1])/l_min(L)*l
    nl = probs_nobs([np.array([lo]+[np.ones_like(lo)]*5)]*4,P_list,CB, n_hard=n_hard,
                    k_C=k_C,k_O=k_O,k_Mg=k_Mg,k_Si=k_Si,
                    k_h2o=k_h2o, 
                    k_atm_source=k_atm_source, k_atm_N=k_atm_N, k_atm_min=k_atm_min,
                    k_SEP=k_SEP,
                    k_guest=k_guest,
                    p_measure=p_measure,
                    return_ps=False)
    Dl = np.sum(nl)
    pl = np.sum(nl[lo>1/1.8])/Dl
    return r3(pl)


############################    
# TEST MULTIPLE HYPOTHESES #
############################
    


Q_avg=0 #caluclates N_obs/<N> if =1
Q_tO2=0 #calculates p(t_O2/t_star) if =1

if 'L5' not in globals():
    L5 = generate_random_samples()

Btt = np.prod([0.151, 0.0537, 0.0976, 0.0873, 0.218])
    
def compute_probs(LB=L5,
                  CB=(1,1),
                  H_photo=[2],
                  H_TL=[0],
                  H_conv=[0],
                  H_bio=[0],
                  H_plates=[0],
                  H_hj=[0],
                  H_terr=[0],
                  H_temp=[0],
                  H_time=[0],
                  H_area=[0],
                  H_S=[1],
                  H_O2=[0],
                  H_death=[0],
                  H_comets=[0],
                  H_grbs=[0],
                  H_glac=[0],
                  H_vol=[0],
                  H_metal=[0],
                  H_CO=[1],
                  H_NPS=[0],
                  H_Fe=[0],
                  H_eccentricity=[0],
                  H_obliquity=[0],
                  H_ocean=[0],
                  H_atm=[0],
                  H_B=[0],
                  H_ool=[0],
                  H_water=[0],
                  H_binary=[0],
                  H_icy_moons=[0],
                  H_rogue=[0],
                  H_guest=[0],
                  Q_l=0,
                  Q_ER=0,
                  n_hard=1,
                  k_C=-.0089,k_O=.0037,k_Mg=-.500,k_Si=.0062,
                  e_max=.25, r_damp=.5, k_h2o=0,
                  k_atm_source=0, k_atm_N=0, k_atm_min=0,
                  k_SEP=1,
                  k_guest=0,
                  p_measure=pmeas,
                  min_prob=-1,
                  max_P=10**10,
                  B0=0.001124497527552,
                  verbose=True,
                  return_df=False,
                  return_Plist=False,
                  use_cache=None):
    """
    LB = generate_random_samples(Ns)
    
    Toggles:
    unless otherwise specified, 0: off, 1: on
    H_photo: photosynthesis 1: photo, 2: yellow
    H_TL: tidal locking
    H_conv: convective stars
    H_bio: biological timescale
    H_plates: plate tectonics
    H_hj: hot jupiters
    H_terr: terrestrial planets 1: giant impact 2: isolation
    H_temp: temperate planets
    H_time: p(life) ~ t
    H_area: p(life) ~ A
    H_S: p(life) ~ 1: S, 2: C, 3: min(S,C)
    H_O2: oxygenation 1: drawdown, 2: drawup 3: both
    H_death: extrinctions 1: setback, 2: IDH, 3: reset
    H_comets: comets
    H_grbs: gamma ray bursts
    H_glac: glaciations
    H_vol: volcanoes
    H_metal: metal to rock ratio
    H_CO: 1: C/O, 2: Mg/Si
    H_NPS: 1: nitrogen, 2: phospohrus, 3: sulfur
    H_Fe: iron
    H_eccentricity: eccentricity
    H_obliquity: obliquity
    H_ocean: ocean, 1: asteroids, 2: grand tack, 3: comets, 4: magma ocean
    H_atm: atmosphere, 1: threshold, 2: slow rotators
    H_B: magnetic field
    H_ool: origin of life, see function for details
    H_water:
    H_binary:
    H_icy_moons:
    H_rogue:
    H_guest: quickly adds another criterion. Toggles 0-1-2
    
    To add a new variable, need to change:
        Hs, H_list, P_primes, and ps in probs_nobs
    
    Q_l: calculates p(M_sun) if =1
    Q_avg: caluclates N_obs/<N> if =1
    Q_tO2: calculates p(t_O2/t_star) if =1
    Q_ER: calculates p(E_R) if =1
    n_hard: number of hard steps
    k_C, k_O, k_Mg, k_Si: threshold energies for these elements
    e_max: maximum allowable eccentricity
    r_damp: smallest undamped asteroid mass
    k_h2o: toggles water habitability conditions
    k_atm_source: source of atmosphere: 0: delivery 1: accretion
    k_atm_N: exponent, how planet atmosphere scales with nitrogen abundance
    k_atm_min: minimum atmosphere: 0: x-ray 1: triple point 2: diurnal
    k_SEP: exponent, distribution of SEP energy
    k_guest: quickly adds another paramter
    p_measure: changes prior measure of universe abundances
    B0: baseline Bayes factor to compare against
    verbose: prints probabilities
    min_prob: only displays row if min(p)>min_prob
    max_P: only calculates probabilities with max_P number of criteria
    return_df: returns results as a dataframe
    use_cache: file path, avoids redoing same computations
    """        
    H_list=[H_photo,H_TL,H_conv,H_bio,\
            H_plates,H_hj,H_terr,H_temp,\
            H_time,H_area,H_S,H_O2,\
            H_death,H_comets,H_grbs,H_glac,H_vol,\
            H_metal,H_CO,H_NPS,H_Fe,\
            H_eccentricity,H_obliquity,H_ocean,\
            H_atm,H_B,\
            H_ool,\
            H_water,H_binary,H_icy_moons,H_rogue,\
            H_guest]
    data=[]
    ids=[]
    P_lists=[]
    start=time.time()
    if use_cache and os.path.exists(use_cache):
        with open(use_cache,'rb') as f:
            cached_ps = pickle.load(f)
    else:
        cached_ps = dict()
    header = ' p(alpha) , p(beta) , p(gamma) '\
            +', p(delta_u) , p(delta_d) '\
            +', p(lambda) '*Q_l\
            +', N_0/<N> '*Q_avg\
            +', p(tO2/tstar)_u , p(tO2/tstar)_m '*Q_tO2\
            +', p(E_R), p(sugar/rock) '*Q_ER
    if verbose:
        print('['+header+']')
    H_all = itertools.product(*H_list)
    if max_P<50:
        N_all = np.array(list(H_all))
        H_all = N_all[np.where(sum(N_all.T>0)<=max_P)]
    if not verbose:
        H_all = tqdm(H_all)
    for P_list in H_all:
        if tuple(P_list) in cached_ps:
            ps = cached_ps[tuple(P_list)]
        else:
            ps = probs_nobs(LB=LB, CB=CB, P_list=P_list, 
                        Q_l=Q_l, Q_ER=Q_ER,
                        n_hard=n_hard,
                        k_C=k_C,k_O=k_O,k_Mg=k_Mg,k_Si=k_Si,
                        k_h2o=k_h2o, e_max=e_max, r_damp=r_damp,
                        k_atm_source=k_atm_source, k_atm_N=k_atm_N, k_atm_min=k_atm_min,
                        k_SEP=k_SEP,
                        k_guest=k_guest,
                        p_measure=p_measure)
            if use_cache:
                cached_ps[tuple(P_list)] = ps
                with open(use_cache,'wb') as f:
                    pickle.dump(cached_ps,f)
        if verbose:
            if min(ps)>min_prob:
                print(signs(P_list))
                print(ps)
            print('bayes factor:',r3(np.prod(ps)/B0))
        P_lists.append(P_list)
        data.append(ps)
        ids.append(signs(P_list))
    end=time.time()
    if verbose:
        print(r2(end-start),'s')
    if return_df:
        cols = [h.strip() for h in header.split(',')]
        df = pd.DataFrame(data,
                          columns=cols,
                          index=ids)
        prod = df.prod(1)
        df['min']=df.apply(min,axis=1)
        df['product']=prod
        if return_Plist:
            df['P_list'] = P_lists
        return df 
