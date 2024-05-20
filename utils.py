import numpy as np

kappa_hat=1

# FUNCTIONS
def heaviside(x):
    return np.heaviside(x,1)
def m1(x):
    return min(x,1-x) 
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
    #return minn(1,x)
    return x*(x<1)+(x>1)
def relu(x):
    return x*heaviside(x)
def rampup(x):
    return heaviside(x)*minn(1,x)
def signs(P_list):
    say=''.join([P_rims[i][P_list[i]] for i in range(len(P_list))])
    if say.strip()=='':
        return 'au naturale'
    else:
        return say

def stamp(L):
    if not isinstance(L,np.ndarray):
        return L
    s = list(L.shape)
    #if len(s)==1:
    #    l = L[0]
    #elif len(s)==2:
    #    l = L[0][0]
    l = np.mean(L)
    if type(l)!=np.float64:
        l = np.mean(l)
    return tuple(s+[l])

def cachewrap(f):
    cachedict=dict()
    def fwrap(*args):
        T=tuple([str(f)]+
            [stamp(k) for k in args])
        T = tuple([t if not isinstance(t,np.ndarray) 
                   else np.mean(t) for t in T])
        if T in cachedict:
            return cachedict[T]
        else:
            O = f(*args)
            cachedict[T] = O
            return O
    return fwrap

P_primes=[['      ',' photo','yellow'],\
          ['   ',' TL',' ~TL'],\
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
          [' ',' nitrogen',' phosphorus',' sulfur'],\
          ['   ',' iron'],\
          ['             ',' eccentricity'],\
          ['          ',' obliquity'],\
          ['                ','  h2o(asteroids)',' h2o(grand tack)',\
           '     h2o(comets)',' h2o(magma ocean)'],\
          ['    ',' atm',' slow rot'],\
          ['  ',' B'],\
          ['',' lightning', ' SEP', ' XUV', ' vents', ' IDP',' comets',
           ' asteroids', ' moneta',' plan pans',' stel pans'],\
          ['',' K/Na/Cl pumps', ' ice floats', ' viscosity increased 23%',
           ' viscosity + ice', ' al26', ' water weight'],\
          ['       ',' binary'],\
          ['          ',' icy moons'],\
          ['      ',' rogue'],\
          ['      ',' guest',' tseug']]

P_rims = [[p[0].strip()]+p[1:] for p in P_primes]