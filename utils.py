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
          ['      ',' guest',' tseug']]

P_rims = [[p[0].strip()]+p[1:] for p in P_primes]