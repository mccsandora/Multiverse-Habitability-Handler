
import itertools
import numpy as np
import sobol_seq




beta_imf=2.35
def l_min(L): #smallest star capable of H fusion
    l,a,b,c,du,dd=tuple(L)
    return .0393*a**(3/2)*b**(-3/4)


def make_random(Ns=10**5):
    E = sobol_seq.i4_sobol_generate(6,Ns)
    return E.T

def generate_random_samples(Ns=10**5):
    E = sobol_seq.i4_sobol_generate(6,Ns).T
    lb = [rescale_all(l) for l in itertools.permutations(E)]
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

def rescale_permute2(number):
    E = make_random(number)
    z = np.zeros((6,0))
    l0,l1,l2,l3 = z,z,z,z
    for l in itertools.permutations(E):
        l = rescale_all(l)
        try:
            l0 = np.concatenate([l0,l[0]],axis=1)
        except:
            continue
        try:
            l1 = np.concatenate([l1,l[1]],axis=1)
        except:
            continue
        try:
            l2 = np.concatenate([l2,l[2]],axis=1)
        except:
            continue
        try:
            l3 = np.concatenate([l3,l[3]],axis=1)
        except:
            continue
    return [l0,l1,l2,l3]

def rescale_permute2(number):
    E = make_random(number)
    z = np.zeros((6,0))
    l0,l1,l2,l3 = z,z,z,z
    for l in itertools.permutations(E):
        l = rescale_all(l)
        l0 = np.concatenate([l0,l[0]],axis=1)
        l1 = np.concatenate([l1,l[1]],axis=1)
        l2 = np.concatenate([l2,l[2]],axis=1)
        l3 = np.concatenate([l3,l[3]],axis=1)
    return [l0,l1,l2,l3]

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

#def rescale_split(E,N=10):
#    lb = [rescale_all(l) for l in np.hsplit(E,N)]
#    l0 = np.concatenate([l[0] for l in lb],axis=1)
#    l1 = np.concatenate([l[1] for l in lb],axis=1)
#    l2 = np.concatenate([l[2] for l in lb],axis=1)
#    l3 = np.concatenate([l[3] for l in lb],axis=1)
#    return [l0,l1,l2,l3]
        
#def make_random_all(Ns=10**5):
#    E = sobol_seq.i4_sobol_generate(6,Ns).T
#    L_normal = rescale_normal(E)
#    L_plates = rescale_plates(E)
#    L_CO = rescale_CO(E)
#    L_plates_CO = rescale_plates_CO(E)
#    LB =[bounds(L_normal), bounds(L_plates),\
#         bounds(L_CO), bounds(L_plates_CO)]
#    print([len(l.T) for l in LB])
#    return LB
        
    
    
    
    