# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:53:21 2020

@author: mccul
"""

import numpy as np
import pandas as pd
import math, time, itertools
from matplotlib import pyplot as plt
from scipy.special import erf, erfc
import sobol_seq


#FUNCTIONS
def heaviside(x):
    return np.heaviside(x,1)
def m1(x):
    return min(x,10-x) 
def r2(p):
    return float('%.2g' % p)
def r3(p):
    return float('%.3g' % p)
def m3(p):
    if p==1.0:
        return 1
    else:
        return float('%.3g' % min(p,1-p))
minn=np.vectorize(min)
def ramp(x):
    return minn(1,x)
def relu(x):
    return x*heaviside(x)
def signs(P_list):
    say=''.join([P_rims[i][P_list[i]] for i in range(len(P_list))])
    if say.strip()=='':
        return 'au naturale'
    else:
        return say



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

###########
#CHEMISTRY#
###########
r_rm=1.9
def f_mr(L,p_metal): #metal to rock ratio
    l,a,b,c,du,dd=tuple(L)
    if p_metal==1:
        return 1-(1+.5*(r_rm*a**(-.945)*b**(-.068)*c**(-.54))**2)**(-.63)
    else:
        return 1

def f_CO(L,p_CO):
    l,a,b,c,du,dd=tuple(L)
    Delta_ER = 1.35*du+2.92*dd+3.97*a-8.24
    if p_CO==1:
        return heaviside(Delta_ER-(-.04))*heaviside(.015-Delta_ER)
    elif p_CO==2:
        return heaviside(.07-Delta_ER-0*(-.11))    
    else:
        return 1

def bounds(L):
    l,a,b,c,du,dd=tuple(L)
    ao = 1/137.036
    bo = .511/938.28
    uo = 2.16/938.28
    do = 4.67/938.28
    # Delta++ unstable
    C_dpp = (0.319734+b*bo-dd*do+du*uo)>0
    # Delta- unstable
    C_dm = (0.319734+b*bo+dd*do-du*uo)>0
    # heavy elements stable
    C_oe = (0.0124696+b*bo-dd*do+du*uo)>0 
    # p stable in nuclei
    C_ps = (0.00884597+b*bo+dd*do-du*uo)>0
    # H stable (e+ emission)
    C_hpe = (-0.177*a*ao + b*bo + dd*do - du*uo)>0
    # H stable (e- capture)
    C_hec = (-0.177*a*ao - b*bo + dd*do - du*uo)>0 
    # pp exothermic
    C_pp = (0.00820651 + 0.177*a*ao - b*bo
            - 1.80527*dd*do + 0.194729*du*uo)>0
    # D stable (weak)
    C_dsw = (0.00820651 + 0.177*a*ao + b*bo 
             - 1.80527*dd*do + 0.194729*du*uo)>0
    # D stable (strong)
    C_dss = (0.00820651 - 0.805271*(dd*do + du*uo) )>0
    # diproton unstable
    C_dp = (-0.00545946 + dd*do + du*uo )>0
    C_pos = (a>0)*(b>0)*(c>0)*(du>0)*(dd>0)
    C_tot = C_dpp*C_dm*C_oe*C_ps*C_hpe*C_hec*C_pp*C_dsw*C_dss*C_dp*C_pos
    return L.T[C_tot].T

def rescale_normal(E):    
    e_l,e_a,e_b,e_c,e_u,e_d=tuple(E)
    a_min=0.2
    a_max=7.89 
    b_min=0.0
    b_max=3.50
    c_min=0.1
    c_max=134
    du_min=0
    du_max=2.21
    dd_min=0.55
    dd_max=2.05    
    a = e_a*(a_max-a_min)+a_min
    b = e_b*(b_max-b_min)+b_min
    c = e_c*(c_max-c_min)+c_min
    du = e_u*(du_max-du_min)+du_min
    dd = e_d*(dd_max-dd_min)+dd_min
    l = l_min([0,a,b,c,0,0])/e_l**(1/(beta_imf-1))
    L = np.array([l,a,b,c,du,dd])
    return L
    
def rescale_plates(E):
    e_l,e_a,e_b,e_c,e_u,e_d=tuple(E)
    a_min=0.64
    a_max=1.41 
    b_min=0.0
    b_max=3.50
    c_min=0.1
    c_max=134
    du_min=.15
    du_max=1.85
    #dd_min=0.63
    #dd_max=1.21    
    a = e_a*(a_max-a_min)+a_min
    b = e_b*(b_max-b_min)+b_min
    c = e_c*(c_max-c_min)+c_min
    du = e_u*(du_max-du_min)+du_min
    dd = .165*e_d+1.47*a-.47*du-.012
    l = l_min([0,a,b,c,0,0])/e_l**(1/(beta_imf-1))
    L = np.array([l,a,b,c,du,dd])
    return L
    
def rescale_CO(E):
    e_l,e_a,e_b,e_c,e_u,e_d=tuple(E)
    a_min=0.35
    a_max=1.30 
    b_min=0.0
    b_max=3.50
    c_min=0.1
    c_max=134
    du_min=0.0
    du_max=2.12
    #dd_min=0.68
    #dd_max=1.10 
    Em=-.5
    Ep=+.5
    a = e_a*(a_max-a_min)+a_min
    b = e_b*(b_max-b_min)+b_min
    c = e_c*(c_max-c_min)+c_min
    du = e_u*(du_max-du_min)+du_min
    dd = 1/2.92*((Ep-Em)*e_d-1.35*du-3.97*a+7.86+Em)
    l = l_min([0,a,b,c,0,0])/e_l**(1/(beta_imf-1))
    L = np.array([l,a,b,c,du,dd])
    return L
    
def rescale_plates_CO(E):
    e_l,e_a,e_b,e_c,e_u,e_d=tuple(E)
    #a_min=0.84
    #a_max=1.02 
    b_min=0.0
    b_max=2.75
    c_min=0.1
    c_max=134
    #du_min=.51
    #du_max=1.41
    #dd_min=0.77
    #dd_max=1.10
    a = .84-.0021*e_d+.122*e_a+.058*e_u
    b = e_b*(b_max-b_min)+b_min
    c = e_c*(c_max-c_min)+c_min
    du = 1.33-.717*e_d+.069*e_a-.03*e_u
    dd = .766+.335*e_d+.147*e_a-.065*e_u
    l = l_min([0,a,b,c,0,0])/e_l**(1/(beta_imf-1))
    L = np.array([l,a,b,c,du,dd])
    return L

def rescale_all(E):
    L_normal = rescale_normal(E)
    L_plates = rescale_plates(E)
    L_CO = rescale_CO(E)
    L_plates_CO = rescale_plates_CO(E)
    LB =[bounds(L_normal), bounds(L_plates),\
         bounds(L_CO), bounds(L_plates_CO)]
    return LB

def rescale_split(E,N=10):
    lb = [rescale_all(l) for l in np.hsplit(E,N)]
    l0 = np.concatenate([l[0] for l in lb],axis=1)
    l1 = np.concatenate([l[1] for l in lb],axis=1)
    l2 = np.concatenate([l[2] for l in lb],axis=1)
    l3 = np.concatenate([l[3] for l in lb],axis=1)
    return [l0,l1,l2,l3]

def rescale_permute(E):
    if type(E)==int:
        E = make_random(E)
    lb = [rescale_all(l) for l in itertools.permutations(E)]
    l0 = np.concatenate([l[0] for l in lb],axis=1)
    l1 = np.concatenate([l[1] for l in lb],axis=1)
    l2 = np.concatenate([l[2] for l in lb],axis=1)
    l3 = np.concatenate([l[3] for l in lb],axis=1)
    return [l0,l1,l2,l3]

def rescale_old(E,include_bounds=False):
    if type(E)==int:
        E = make_random(E)
    e_l,e_a,e_b,e_c,e_u,e_d=tuple(E)
    a_min=0.2
    a_ptmax=137/136
    a_ptmin=137/153
    a_max=2.07 #-pt
    b_min=0.0
    b_max=1.1/.511 #+pp
    c_min=0.1
    c_max=134
    a = e_a*(a_max-a_min)+a_min
    ap = e_a*(a_ptmax-a_ptmin)+a_ptmin
    b = e_b*(b_max-b_min)+b_min
    c = e_c*(c_max-c_min)+c_min
    d = np.full(len(a),1)
    l = l_min([0,a,b,c,0,0])/e_l**(1/(beta_imf-1))
    lp = l_min([0,ap,b,c,0,0])/e_l**(1/(beta_imf-1))
    L = np.array([l,a,b,c,d,d])
    Lp = np.array([lp,ap,b,c,d,d])
    #Lo = np.full((6,10),0.5)
    if include_bounds:
        return [bounds(L), bounds(Lp)]
    else:
        return [L, Lp]

def make_random(Ns=10**5):
    E = sobol_seq.i4_sobol_generate(6,Ns)
    return E.T

def make_random_all(Ns=10**5):
    E = sobol_seq.i4_sobol_generate(6,Ns).T
    L_normal = rescale_normal(E)
    L_plates = rescale_plates(E)
    L_CO = rescale_CO(E)
    L_plates_CO = rescale_plates_CO(E)
    LB =[bounds(L_normal), bounds(L_plates),\
         bounds(L_CO), bounds(L_plates_CO)]
    print([len(l.T) for l in LB])
    return LB

def estimate_error(df1,df2):
    a1=np.array(df1.drop(columns=['P_list']))
    a2=np.array(df2.drop(columns=['P_list']))
    A = 2*abs(a1-a2)/(a1+a2)
    return A
    #return -np.sort(-A.reshape(-1))
    
#####################
#NUMBER OF OBSERVERS#
#####################
    
def nobs(L,P_list): #counts number of observers
    p_photo,p_TL,p_conv,p_bio,p_plates,\
    p_hj,p_terr,p_temp,p_time,p_area,p_S,\
    p_O2,p_death,p_comets,p_grb,p_glac,p_vol,\
    p_metal,p_CO,p_guest = P_list

    #L = bounds(L)
    
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
    *r_time(L, p_time)\
    *r_area(L, p_area)\
    *S_tot(L, p_S)\
    *f_O2(L, p_O2)\
    *f_int(L, p_death, p_comets, p_grb, p_glac, p_vol)\
    *f_mr(L, p_metal)\
    *f_CO(L, p_CO)
    
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

def probs_nobs(LB, P_list, Q_l=1, Q_ER=0, n_hard=1):
    p_photo,p_TL,p_conv,p_bio,p_plates,\
    p_hj,p_terr,p_temp,p_time,p_area,p_S,\
    p_O2,p_death,p_comets,p_grb,p_glac,p_vol,\
    p_metal,p_CO,p_guest = P_list

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
        *f_CO(L, p_CO)
    
    D = sum(n)
    pa = sum(n[a>=1])/D
    pb = sum(n[b>=1])/D
    pc = sum(n[c>=1])/D
    pdu = sum(n[du>=1])/D
    pdd = sum(n[dd>=1])/D
    pall = [m3(pa),m3(pb),m3(pc),m3(pdu),m3(pdd)]
    if Q_l==1:
        pl = probs_l(L,P_list)
        pall.append(pl)
    if Q_ER==1:
        Delta_ER = 1.35*du+2.92*dd+3.97*a-8.24
        pER = sum(n[Delta_ER>0])/D
        pall.append(m3(pER))        
    return pall

def probs_l(L, P_list): #probability of being around a sunlike star within our universe
    l,a,b,c,du,dd=tuple(L)    
    lo = l_min([1,1,1,1,1,1])/l_min(L)*l
    nl = nobs([lo,1,1,1,1,1],P_list)
    Dl = sum(nl)
    pl = sum(nl[lo>1/1.8])/Dl
    return r3(pl)
 
# a=alpha/alpha_obs, b=beta/beta_obs, c=gamma/gamma_obs, l=lambda
#a_min=0.2
#a_max=7.89 #without plate tectonics
##a_ptmin=137/153
##a_ptmax=137/136 #with plate tectonics
#b_min=0.0
#b_max=3.50
#c_min=0.1
#c_max=134
#du_min=0
#du_max=2.21
#dd_min=0.55
#dd_max=2.05    

#def make_sobol(Ns=10**5,p_quarks=1):
#    try:
#        E = sobol_seq.i4_sobol_generate(6,Ns)
#    except:
#        print('pip install sobol_seq for better accuracy')
#        E = np.random.random_sample((6,Ns))
#    at = E[:,1]*(a_max-a_min)+a_min
#    #ap = E[:,1]*(a_ptmax-a_ptmin)+a_ptmin
#    bt = E[:,2]*(b_max-b_min)+b_min
#    ct = E[:,3]*(c_max-c_min)+c_min
#    lt = l_min([0,at,bt,ct,0,0])/E[:,0]**(1/(beta_imf-1))
#    #ltp = l_min(ap,bt,ct)/E[:,0]**(1/(beta_imf-1))
#    #lo = l_min(1,1,1)/E[:,0]**(1/(beta_imf-1))
#    dut = (E[:,4]*(du_max-du_min)+du_min)**p_quarks
#    ddt = (E[:,5]*(dd_max-dd_min)+dd_min)**p_quarks
#    L = np.array([lt,at,bt,ct,dut,ddt])
#    return L
    



#def truss_rand(E,p_quarks=1):    
#    a_min=0.2
#    a_max=7.89 #without plate tectonics
#    #a_ptmin=137/153
#    #a_ptmax=137/136 #with plate tectonics
#    b_min=0.0
#    b_max=3.50
#    c_min=0.1
#    c_max=134
#    du_min=0
#    du_max=2.21
#    dd_min=0.55
#    dd_max=2.05
#    at = E[:,1]*(a_max-a_min)+a_min
#    #ap = E[:,1]*(a_ptmax-a_ptmin)+a_ptmin
#    bt = E[:,2]*(b_max-b_min)+b_min
#    ct = E[:,3]*(c_max-c_min)+c_min
#    lt = l_min([0,at,bt,ct,0,0])/E[:,0]**(1/(beta_imf-1))
#    #ltp = l_min(ap,bt,ct)/E[:,0]**(1/(beta_imf-1))
#    #lo = l_min(1,1,1)/E[:,0]**(1/(beta_imf-1))
#    dut = (E[:,4]*(du_max-du_min)+du_min)**p_quarks
#    ddt = (E[:,5]*(dd_max-dd_min)+dd_min)**p_quarks
#    L = np.array([lt,at,bt,ct,dut,ddt])
#    return L

#####################################     
#TEST MULTIPLE HYPOTHESES ON RUNNING#
#####################################
    
#number of samples. 10**5 is decent, 10**6 is accurate, 10**7 professional

Q_l=1 #calculates p(M_sun) if =1
Q_avg=0 #caluclates N_obs/<N> if =1
Q_tO2=0 #calculates p(t_O2/t_star) if =1
Q_ER=0

P_primes=[['      ',' photo','yellow'],\
          ['   ',' TL'],\
          ['     ',' conv'],\
          ['    ',' bio'],\
          ['       ',' plates'],\
          ['   ',' HJ','hj-pd'],\
          ['    ','  GI','  iso'],\
          ['     ',' temp'],\
          ['     ',' time'],\
          ['     ',' area'],\
          ['  ',' S',' C',' min(S,C)'],\
          ['       ',' O2down','   O2up',' O2both'],\
          ['        ',' setback','     IDH','   reset'],\
          ['       ',' comets',],\
          ['     ',' grbs'],\
          ['     ',' glac'],\
          ['    ',' vol'],\
          ['      ',' metal'],\
          ['    ',' C/O',' Mg/Si'],\
          ['      ',' guest',' tseug']]

P_rims = [[p[0].strip()]+p[1:] for p in P_primes]
P_yS = (2, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
H_mega = [[0,1,2],[0,1],[0,1],[0,1],[0,1],[0,1],\
          [0,1,2],[0,1],[0,1],[0,1],[0,1,2,3],[0,1,2,3],\
          [0,1,2,3],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1,2]]

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
              H_S=[0,1],
              H_O2=[0],
              H_death=[0],
              H_comets=[0],
              H_grbs=[0],
              H_glac=[0],
              H_vol=[0],
              H_metal=[0],
              H_CO=[0],
              H_guest=[0],
              Q_l=1,
              Q_ER=0,
              n_hard=1,
              verbose=False,
              min_prob=-1,
              max_P=10**10,
              return_df=True):
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
            H_metal,H_CO,H_guest]
    data=[]
    ids=[]
    P_lists=[]
    start=time.time()
    header = ' p(alpha) , p(beta) , p(gamma) '\
            +', p(delta_u) , p(delta_d) '\
            +', p(lambda) '*Q_l\
            +', N_0/<N> '*Q_avg\
            +', p(tO2/tstar)_u , p(tO2/tstar)_m '*Q_tO2\
            +', p(E_R) '*Q_ER
    if verbose:
        print('['+header+']')
    H_all = itertools.product(*H_list)
    if max_P<50:
        N_all = np.array(list(H_all))
        H_all = N_all[np.where(sum(N_all.T>0)<=max_P)]
    for P_list in H_all:
    #    if sum(np.array(P_list)>0)<=max_P:
        ps = probs_nobs(LB, P_list, Q_l, Q_ER, n_hard)
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
        df = pd.DataFrame(data,
                          columns=[h.strip() for h in header.split(',')],
                          index=ids)
        #df['P_list'] = P_lists
        return df 

    




