
import numpy as np
import pandas as pd
import math, time, itertools
from scipy.special import erf, erfc, gammaincinv
from scipy.integrate import quad  
from sample_generator import *
#from utils import *
#from useful_functions import *
from utils import heaviside, minn, relu, ramp, signs, m3, r3, r2, P_primes, P_rims


#######           
#STARS#
#######
        
def pmeas(L): #theory measure
    l,a,b,c,du,dd=tuple(L)
    return 1/b/c/du/dd
def Nstars(L): #number of stars in the universe
    l,a,b,c,du,dd=tuple(L)
    return a**(-3/2)*b**(3/4)*c**3

beta_imf=2.35
def l_min(L): #smallest star capable of H fusion
    l,a,b,c,du,dd=tuple(L)
    return .0393*a**(3/2)*b**(-3/4)
#def p_lmin(L): #Salpeter initial mass function
#    l,a,b,c,du,dd=tuple(L)
#    l_minus=.0393*a_min**(3/2)*b_max**(-3/4)
#    return (l_min(L)/l_minus)**(beta_imf-1)*heaviside(l-l_min(L))
def masch(L): # gives imf a knee
   l,a,b,c,du,dd=tuple(L)
   return 8.9403*(1+(5.06*l_min(L)/l)**1.35)**-1.4

#NUCLEAR EFFICIENCY
def e_nuc(L):
    l,a,b,c,du,dd=tuple(L)
    return .84-.03*a+.19*np.sqrt(.32*du+.68*dd)

#TIDAL LOCKING
def l_TL(L): #tidal locking limit
    l,a,b,c,du,dd=tuple(L)
    return .474*a**(5/2)*b**(1/2)*c**(-4/11)*e_nuc(L)**(1/11)
def l_TLb(L): #t_TL compared to t_bio
    l,a,b,c,du,dd=tuple(L)
    return .474*a**(47/17)*b**(12/17)*c**(-4/11)
def f_TL(L, p_TL):
    l,a,b,c,du,dd=tuple(L)
    if p_TL==1:
        return heaviside(l-l_TL(L))
    else:
        return 1

#CONVECTION
def l_conv(L): #convective stellar mass
    l,a,b,c,du,dd=tuple(L)
    return .194*a**3*b*c**(-1/2)
def f_conv(L, p_conv):
    l,a,b,c,du,dd=tuple(L)
    if p_conv==1:
        return heaviside(l-l_conv(L))
    else: 
        return 1

#BIOLOGICAL TIMESCALE
def l_bio(L): #stellar lifetime is Gyr
    l,a,b,c,du,dd=tuple(L)
    return 1.2*a**(8/5)*b**(-1/5)*c**(-4/5)*e_nuc(L)**(2/5)
def f_bio(L, p_bio):  
    l,a,b,c,du,dd=tuple(L)
    if p_bio==1:
        return heaviside(l_bio(L)-l)
    else:
        return 1

#PHOTOSYNTHESIS
def l_fizzle(L,w): #star too red: w is longest wavelength in nm
    l,a,b,c,du,dd=tuple(L)
    return .21*(w/1100)**(-40/19)*a**(60/19)*b**(20/19)*c**(-10/19)
def l_fry(L,w): #star too blue: w is shortest wavelength in nm
    l,a,b,c,du,dd=tuple(L)
    return 1.75*(w/400)**(-40/19)*a**(60/19)*b**(20/19)*c**(-10/19)
def f_photo(L, p_photo): 
    l,a,b,c,du,dd=tuple(L)
    if p_photo==2:
        return heaviside(l-l_fizzle(L,750))*heaviside(l_fry(L,600)-l)
    elif p_photo==1:
        return heaviside(l-l_fizzle(L,1100))*heaviside(l_fry(L,400)-l)
    else:
        return 1


################################
#FRACTION OF STARS WITH PLANETS#
################################
    
Z_inf=.011
def Z_min(L): #minimum metallicity for planet formation
    l,a,b,c,du,dd=tuple(L)
    return 6.3*10**-4*(1.8*l)**(3/4)*a**-3*b**(-1/2)*c**(1/2)
def f_pstep(L): #fraction of stars with planets (step function)
    return heaviside(Z_inf-Z_min(L))
def f_sn(L): #fraction of galaxies that can retain sn ejecta
    l,a,b,c,du,dd=tuple(L)
    return erfc(4.41*10**(-5)*a**2*b**(5/3))
def f_p(L): #fraction of stars with planets
    return f_sn(L)*f_pstep(L)

#HOT JUPITERS
def Z_maxpp(L): #maximum metallicity for no hot jupiters (planet-planet interactions)
    l,a,b,c,du,dd=tuple(L)
    return .12/(1.8*l)*a**(13/6)*b**(-3/2)
def f_hj(L, p_hj): #simplified fraction of earths without hot jupiters
    if p_hj==1:
        return (1-(Z_inf/Z_maxpp(L))**2)*heaviside(Z_maxpp(L)-Z_inf)
    else:
        return 1


#############################
#NUMBER OF HABITABLE PLANETS#
#############################

#ISOLATION MASS
qiso=1/3 #slope of planetesimal masses
p_sigma=0 # planet variance set by 0: entropy or 1: shot noise
p_ldep=1 # include stellar mass-planet mass correlation 0: no 1: yes
def r_iso(L): #ratio of isolation mass to terrrestrial mass
    l,a,b,c,du,dd=tuple(L)
    return .1*(1.8*l)**2*a**-4*b**(-21/8)
def r_inn(L): #ratio of innder disk mass to isolation mass
    l,a,b,c,du,dd=tuple(L)
    return 30*(1.8*l)**(-1/3)*a**(5/6)*b**(5/8)
def fiso(x): #fraction of terrestrial planets
    return min(1,(x/.3)**qiso)-min(1,(x/4)**qiso)
f_iso=np.vectorize(fiso)
def n_iso(x): # average number of planets
    return (qiso-1)/qiso*(x-x**(1-qiso))/(1-x**(1-qiso))

#GIANT IMPACT MASS
def r_olig(L): #ratio of oligarch mass to terrestrial mass
    l,a,b,c,du,dd=tuple(L)
    return 2.64*(1.8*l)**(p_ldep*5/2)*a**(-9/2)*b**(-45/16) #accretion
#    return 2.64*(1.8*l)**(p_ldep*139/40)*a**(-39/4)*b**(-3)*c**(3/4) #irradiation
def f_olig(x,s): #number of terrestrial mass oligarchs
    return 1/2*(erf((np.log(4/x)+s**2/2)/np.sqrt(2)/s)\
                -erf((np.log(.3/x)+s**2/2)/np.sqrt(2)/s))
#    return np.exp(-np.pi/4*(.3/x)**2)-np.exp(-np.pi/4*(4/x)**2) #rayleigh dist
def sigma_m(L): #variance of m_olig
    return [1/np.sqrt(6),1/np.sqrt(r_inn(L))]
def n_olig(L): #average number of planets 
    l,a,b,c,du,dd=tuple(L)
    return 3*(1.8*l)**(-5/6)*a**(4/3)*b**(13/16) #accretion
#    return 3*(1.8*l)**(-11/8)*a**(17/4)*b**(11/12)*c**(-5/12) #irradiation

def n_terr(L, p_terr):
    if p_terr==2:
        return n_iso(r_inn(L))*f_iso(r_iso(L))
    elif p_terr==1:
        return n_olig(L)*f_olig(r_olig(L),sigma_m(L)[p_sigma])
    else:
        return 1

#TEMPERATE ZONE
def temp_thresh(L): #temperate zone smaller than disk size
    l,a,b,c,du,dd=tuple(L)
    return .01*(1.8*l)**(17/12)*a**(-5)*b**(-2)*c**(1/2)
def n_temp(L, p_temp): #number of planets in temperate zone
    l,a,b,c,du,dd=tuple(L)
    if p_temp==1:
        return .431*l**(-85/48)*a**(11/2)*b**(7/4)*c**(-5/8)\
    *heaviside(1-temp_thresh(L))
    else:
        return 1


######
#LIFE#
######
    
#n_hard=1 #number of hard steps
def r_time(L, p_time): #ratio of stellar to molecular timescale
    l,a,b,c,du,dd=tuple(L)
    if p_time==1:
        return l**(-5/2)*a**(4)*b**(-1/2)*c**-2*e_nuc(L)
    else:
        return 1
def r_area(L, p_area): #ratio of planet area to molecular area
    l,a,b,c,du,dd=tuple(L)
    if p_area==1:
        return a**(3/2)*b**(3/4)*c**-3 #L_mol~a_0
    else:
        return 1
def S_tot(L, p_S, ec=1): #entropy produced in stellar lifetime
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
def C_tot(L): #material limited biosphere
    l,a,b,c,du,dd=tuple(L)
    return (1.8*l)**(-5/2)*a**(9/2)*b**(1/2)*c**(-3)*e_nuc(L)
def f_plates(L, p_plates):
    l,a,b,c,du,dd=tuple(L)
    if p_plates==1:
        return 1
        #return heaviside(a-.32*du-.68*dd+.104)*\
        #       heaviside(.008-a+.32*du+.68*dd)
        #return heaviside(a_ptmax-a)*heaviside(a-a_ptmin)
    else:
        return 1

#OXYGENATION TIME
r_o=.22 #Earth's ratio
def rip(x): #regulates the exp in rat_t
    return 500*np.tanh(x/500)
def rat_tup(L): #ratio of oxidation time to stellar lifetime
    l,a,b,c,du,dd=tuple(L)
    return r_o*(1.8*l)**(81/40)*a**(-4)*b**(3/4)*c**(3/4)*\
    np.exp(rip(-18.85*(b**(-1/2)-1)\
               +15.37*((1.8*l)**(-19/40)*a**(3/2)*b*c**(-1/4)-1)))/e_nuc(L)
def rat_tdown(L): #ratio of timescales with drawdown
    l,a,b,c,du,dd=tuple(L)
    return r_o*(1.8*l)**(5/2)*a**-3*b**(1/4)\
               *np.exp(18.85*(1-b**(-1/2)))/e_nuc(L)
#def rat_t(L, p_O2):
#    l,a,b,c,du,dd=tuple(L)
#    return [0,rat_tdown(L),rat_tup(L),\
#            2/(1/(rat_tdown(L)+10**-6)+1/rat_tup(L))]
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

#######
#DEATH#
#######
    
Myr=1
km=1
TW=1
def t_rec(L): #recovery time
    l,a,b,c,du,dd=tuple(L)
#    return 10*a**(-2)*b**(-3/2)*Myr #mol
    return 10*(1.8*l)**(17/8)*a**(-15/2)*b**(-3)*c**(-1/4)*Myr #yr
def t_star(L): #stellar lifetime
    l,a,b,c,du,dd=tuple(L)
    return 5500*(1.8*l)**(-5/2)*a**2*b**(-2)

#COMETS
def d_comet(L): #typical comet size
    l,a,b,c,du,dd=tuple(L)
    return 1*(1.8*l)*a**(-25/9)*b**(-3/2)*c**(-5/6)*km
def d_sox(L): #extinction size (sulfates)
    l,a,b,c,du,dd=tuple(L)
    return 1*(1.8*l)**(1/4)*a**(-3)*b**(-1)*c**(-1/2)*km
def d_dust(L): #extinction size (dust)
    l,a,b,c,du,dd=tuple(L)
    return 1*(1.8*l)**(1/4)*a**(-5/3)*b**(-1)*c**(-1/2)/\
(1/2+1/2*erfc(.57*(.769+np.log(.01*a**(1/2)*b**(1/4)*c**(-1)))))**(1/3)*km
def d_both(L):
    l,a,b,c,du,dd=tuple(L)
    return minn(d_sox(L),d_dust(L))
def G_comets(L): #comet extinction rate
    l,a,b,c,du,dd=tuple(L)
    return (1.8*l)**(-16/5)*a**8*b**(-1/2)*(ramp(d_comet(L)/d_both(L)))**(1.5)/(3*Myr)/30

#GRBS
def vh(x): #volume function
    if x<=.075:
        return 40/3*x**3
    if .075<x<1:
        return x**2
    if 1<=x:
        return 1
vg=np.vectorize(vh)
def rat_grb(L): #ratio of grb death radius to galaxy radius 
    l,a,b,c,du,dd=tuple(L)
    return 1/6*a**(-1)*b**(-3/2)*c**(-1/2)
def G_grb(L): #grb extinction rate
    l,a,b,c,du,dd=tuple(L)
    return c*36*vg(rat_grb(L))/(90*Myr)

#GLACIATIONS
def Q_dirty(L): #dimensional analysis estimate of heat flux 
    l,a,b,c,du,dd=tuple(L)
    return 47*a**(9/2)*b**(7/2)*c**(-1)*TW
def s_t(L): #dimensionless quantity in heat flux
    l,a,b,c,du,dd=tuple(L)
    return .4242*a**(1/2)*b**(5/8)*c*(t_rec(L)/Myr)**(1/2)
def Q_form(L): #heat of formation
    l,a,b,c,du,dd=tuple(L)
    return Q_dirty(L)*5.13/s_t(L)*math.e**(-s_t(L))
def f_rad(L): # dimensionless fraction in radioactive heat
    l,a,b,c,du,dd=tuple(L)
    return math.e**(1+52.72*(1-(137/144/a)**(1/2))\
                    -math.e**(52.72*(1-(137/144/a)**(1/2))))
def Q_rad(L): #radioactive heat
    l,a,b,c,du,dd=tuple(L)
    return 47*a**(3/2)*b**(3/4)*c**(-3)*f_rad(L)*TW
def Q_both(L):
    l,a,b,c,du,dd=tuple(L)
    return (Q_form(L)+Q_rad(L))/2
def G_glac(L): #glaciation extinction rate
    l,a,b,c,du,dd=tuple(L)
    return Q_both(L)/(47*TW)*a**(-7/2)*b**(-9/4)*c**3/(90*Myr)

#VOLCANOES
def G_vol(L): #volcano extinction rate
    l,a,b,c,du,dd=tuple(L)
    return (Q_both(L)/(47*TW))**(1/2)*a**(-3/4)*b**(-3/8)*c**(3/2)/(90*Myr)

#TOTAL
def G_death(L, p_comets, p_grb, p_glac, p_vol): #total extinction rate
    l,a,b,c,du,dd=tuple(L)
    return (p_comets*G_comets(L)+p_grb*G_grb(L)\
            +p_glac*G_glac(L)+p_vol*G_vol(L))/(p_comets+p_grb+p_glac+p_vol+10**-6)
def f_setback(t,G):
    return (1-t*G)*heaviside(1-t*G)
def f_IDH(t,G):
    return (t*G+10**-6)*(1-t*G)*heaviside(1-t*G)
def f_reset(t,ts,G):
    return 1-(1-np.exp(-10*t*G))**(ts/(10*t))
def f_int(L, p_death, p_comets, p_grb, p_glac, p_vol): #fraction of life that develops intelligence
    l,a,b,c,du,dd=tuple(L)
    if p_death==1:
        return f_setback(t_rec(L),G_death(L, p_comets, p_grb, p_glac, p_vol))
    elif p_death==2:
        return f_IDH(t_rec(L),G_death(L, p_comets, p_grb, p_glac, p_vol))
    elif p_death==3:
        return f_reset(t_rec(L),t_star(L),\
                       G_death(L, p_comets, p_grb, p_glac, p_vol))
    else:
        return 1

##########
#ELEMENTS#
##########
        
r_rm=1.9
if 'gamma_inv' not in globals():
    gamma_inv = np.array([[quad(lambda x: np.exp(x)*x**(.54-1),0,i)[0],i] 
                     for i in np.arange(.01,20,.001)])


def gi(x):
    a  = np.searchsorted(gamma_inv[:,0],x)
    return gamma_inv[a,1]
def f_mr(L,p_metal): #metal to rock ratio
    l,a,b,c,du,dd=tuple(L)
    if p_metal==1:
        return 1-np.exp(-gi(.638*r_rm*a**(.56)*b**(-.82)*c**(-.54)))
    else:
        return 1

def f_CO(L,p_CO,k_C=-.0089,k_O=.0118,k_Mg=-.500,k_Si=.0062):
    l,a,b,c,du,dd=tuple(L)
    Delta_ER = 1.35*du+2.92*dd+3.97*a-8.24
    if p_CO==1:
        return heaviside(Delta_ER-k_C)*heaviside(k_O-Delta_ER)
    elif p_CO==2:
        return heaviside(Delta_ER-k_Mg)*heaviside(k_Si-Delta_ER)
    else:
        return 1
    
def f_NPS(L,p_NPS):    
    l,a,b,c,du,dd=tuple(L)
    P31_stable = 8.25589*a+0.511*b-4.67*dd+2.16*du
    S32_stable = -5.025+7.95682*a-0.511*b-4.67*dd+2.16*du
    P32_stable = -(-6.575+7.50703*a-0.511*b-4.67*dd+2.16*du)
    #N14_stable = -1.77143+4.76494*a-0.511*b-4.67*dd+2.16*du
    N14_stable = -1.58793+4.76494*a-0.511*b-4.67*dd+2.16*du
    Cl35_stable = -(-5.30286+8.19482*a-0.511*b-4.67*dd+2.16*du)
    if p_NPS==1:
        return heaviside(N14_stable)
    if p_NPS==2:
        return heaviside(P31_stable)+\
               51*heaviside(-S32_stable)*heaviside(P32_stable)+\
                .4*heaviside(-P32_stable)
    elif p_NPS==3: 
        return heaviside(S32_stable)+\
               heaviside(-P31_stable)+heaviside(Cl35_stable)
    else:
        return 1

def f_Fe(L,p_Fe):
    l,a,b,c,du,dd=tuple(L)
    Fe_stable = -(-9.5+10.5412*a-0.511*b-4.67*dd+2.16*du)
    Co_unstable = -3.75714+10.9144*a-0.511*b-4.67*dd+2.16*du
    Ni_unstable = -2.87143+11.2876*a-0.511*b-4.67*dd+2.16*du
    not_Cr = -(-10.3857+10.1679*a-0.511*b-4.67*dd+2.16*du)
    if p_Fe==1:
#        return heaviside(Ni_unstable)*heaviside(Fe_stable)
        return heaviside(Co_unstable)*heaviside(Fe_stable)
#        return heaviside(not_Cr)
#        return heaviside(Co_unstable) 
    else:
        return 1
        
def f_water(L,p_water):
        return 1


#####################
#NUMBER OF OBSERVERS#
#####################
    
def nobs(L,P_list, n_hard=1): #counts number of observers
    p_photo,p_TL,p_conv,p_bio,p_plates,\
    p_hj,p_terr,p_temp,p_time,p_area,p_S,\
    p_O2,p_death,p_comets,p_grb,p_glac,p_vol,\
    p_metal,p_CO,p_NPS,p_Fe,p_guest = P_list

    k_C=-.0089
    k_O=.0118
    k_Mg=-.500
    k_Si=.0062
    return pmeas(L)\
           *Nstars(L)\
           *masch(L)**1\
           *f_photo(L, p_photo)\
           *f_TL(L, p_TL)\
           *f_conv(L, p_conv)\
           *f_bio(L, p_bio)\
           *f_plates(L, p_plates)\
           *f_p(L)**1\
           *f_hj(L, p_hj)\
           *n_terr(L, p_terr)\
           *n_temp(L, p_temp)\
           *r_time(L, p_time)**n_hard\
           *r_area(L, p_area)**n_hard\
           *S_tot(L, p_S)**n_hard\
           *f_O2(L, p_O2)\
           *f_int(L, p_death, p_comets, p_grb, p_glac, p_vol)\
           *f_mr(L, p_metal)\
           *f_CO(L, p_CO, k_C, k_O, k_Mg, k_Si)\
           *f_NPS(L, p_NPS)\
           *f_Fe(L, p_Fe)\
           *f_water(L, p_guest)
    
def probs(L, P_list):
    n = nobs(L,P_list)  
    l,a,b,c,du,dd=tuple(L)
    D = sum(n)
    pa = sum(n[a>=1])/D
    pb = sum(n[b>=1])/D
    pc = sum(n[c>=1])/D
    pdu = sum(n[du>=1])/D
    pdd = sum(n[dd>=1])/D
    pl = probs_l(L,P_list)
    return [m3(pa),m3(pb),m3(pc),m3(pdu),m3(pdd),pl]

def probs_nobs(LB, P_list, Q_l=1, Q_ER=0, n_hard=1,
               k_C=-.0089,k_O=.0118,k_Mg=-.500,k_Si=.0062):
    p_photo,p_TL,p_conv,p_bio,p_plates,\
    p_hj,p_terr,p_temp,p_time,p_area,p_S,\
    p_O2,p_death,p_comets,p_grb,p_glac,p_vol,\
    p_metal,p_CO,p_NPS,p_Fe,p_guest = P_list

    L = LB[p_plates+2*(p_CO>0)]
    l,a,b,c,du,dd=tuple(L)

    n = pmeas(L)\
        *Nstars(L)\
        *masch(L)**1\
        *f_photo(L, p_photo)\
        *f_TL(L, p_TL)\
        *f_conv(L, p_conv)\
        *f_bio(L, p_bio)\
        *f_plates(L, p_plates)\
        *f_p(L)**1\
        *f_hj(L, p_hj)\
        *n_terr(L, p_terr)\
        *n_temp(L, p_temp)\
        *r_time(L, p_time)**n_hard\
        *r_area(L, p_area)**n_hard\
        *S_tot(L, p_S)**n_hard\
        *f_O2(L, p_O2)\
        *f_int(L, p_death, p_comets, p_grb, p_glac, p_vol)\
        *f_mr(L, p_metal)\
        *f_CO(L, p_CO, k_C, k_O, k_Mg, k_Si)\
        *f_NPS(L, p_NPS)\
        *f_Fe(L, p_Fe)\
        *f_water(L, p_guest)
    
    D = sum(n)
    pa = sum(n[a>=1])/D
    pb = sum(n[b>=1])/D
    pc = sum(n[c>=1])/D
    pdu = sum(n[du>=1])/D
    pdd = sum(n[dd>=1])/D
    pall = [m3(pa),m3(pb),m3(pc),m3(pdu),m3(pdd)]
    if Q_l==1:
        pl = probs_l(L,P_list,n_hard)
        pall.append(pl)
    if Q_ER==1:
        Delta_ER = 1.35*du+2.92*dd+3.97*a-8.24
        pER = sum(n[Delta_ER>0])/D
        pER2 = sum(n[(Delta_ER>0) & (Delta_ER<.0035)])/D
        pall.append(m3(pER))     
        pall.append(m3(pER2))
    return pall

def probs_l(L, P_list, n_hard=1): #probability of being around a sunlike star within our universe
    l,a,b,c,du,dd=tuple(L)    
    lo = l_min([1,1,1,1,1,1])/l_min(L)*l
    nl = nobs([lo,1,1,1,1,1],P_list,n_hard)
    Dl = sum(nl)
    pl = sum(nl[lo>1/1.8])/Dl
    return r3(pl)
 



##########################    
#TEST MULTIPLE HYPOTHESES#
##########################
    
#number of samples. 10**5 is decent, 10**6 is accurate, 10**7 professional

#Q_l=1 #calculates p(M_sun) if =1
Q_avg=0 #caluclates N_obs/<N> if =1
Q_tO2=0 #calculates p(t_O2/t_star) if =1
#Q_ER=0

def compute_probs(LB,
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
                  H_CO=[0],
                  H_NPS=[0],
                  H_Fe=[0],
                  H_guest=[0],
                  Q_l=0,
                  Q_ER=0,
                  n_hard=1,
                  k_C=-.0089,k_O=.0118,k_Mg=-.500,k_Si=.0062,
                  min_prob=-1,
                  max_P=10**10,
                  verbose=True,
                  return_df=False):
    """
    LB = make_sobol(Ns)
    
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
    H_guest: quickly adds another criterion. Toggles 0-1-2
    
    To add a new variable, need to change:
        Hs, H_list, P_primes, and ps in probs_nobs
    
    Q_l: calculates p(M_sun) if =1
    Q_avg: caluclates N_obs/<N> if =1
    Q_tO2: calculates p(t_O2/t_star) if =1
    Q_ER: calculates p(E_R) if =1
    n_hard: number of hard steps
    verbose: prints probabilities
    min_prob: only displays row if min(p)>min_prob
    max_P: only calculates probabilities with max_P number of criteria
    return_df: returns results as a dataframe
    """        
    H_list=[H_photo,H_TL,H_conv,H_bio,\
            H_plates,H_hj,H_terr,H_temp,\
            H_time,H_area,H_S,H_O2,\
            H_death,H_comets,H_grbs,H_glac,H_vol,\
            H_metal,H_CO,H_NPS,H_Fe,H_guest]
    data=[]
    ids=[]
    P_lists=[]
    start=time.time()
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
    for P_list in H_all:
        ps = probs_nobs(LB=LB, P_list=P_list, 
                        Q_l=Q_l, Q_ER=Q_ER,
                        n_hard=n_hard,
                        k_C=k_C,k_O=k_O,k_Mg=k_Mg,k_Si=k_Si)
        if verbose:
            if min(ps)>min_prob:
                print(signs(P_list))
                print(ps)
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
        df['min']=df.apply(min,axis=1)
        #df['P_list'] = P_lists
        return df 


