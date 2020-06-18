

import numpy as np



#FUNCTIONS
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
    return minn(1,x)
def relu(x):
    return x*heaviside(x)
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
          ['      ',' guest',' tseug']]

P_rims = [[p[0].strip()]+p[1:] for p in P_primes]