Code to compare multiple habitability hypoetheses in the multiverse framework.  This computes the probability of observing five fundamental constants:
alpha = e^2/(4pi), 
beta = m_electron/m_proton,
gamma = m_proton/M_planck,
delta_u = m_up/m_proton,
delta_d = m_down/m_proton, 
and optionally the probability of orbiting a star as massive as ours, the probability of observing such a small Hoyle resonance energy, and the probability of observing an organic-to-rock ratio as large as ours.  The code currently considers 27 different habitability conditions, amounting to 224x10^9 possible combinations.

To initialize, run

`from muhaha import *`

This may take a minute.

Compute the probabilities for various habitability hypotheses by

`compute_probs()`

By default it's set to yellow + entropy + C/O conditions, but these can be toggled.  For instance:

`compute_probs(H_TL=[0,1], H_terr=[1])`

See muhaha_demo.ipynb for more examples.