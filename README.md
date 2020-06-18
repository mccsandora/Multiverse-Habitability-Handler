To generate the pseudo-random numbers we use the package sobol-seq.  This can be installed by:

pip install sobol-seq

To initialize, in an environment run

from muhaha import *

Generate the random sample by

L5 = generate_random_samples(10**5)

This should take 1-2 minutes to make.

Compute the probabilities for various habitability hypotheses by

compute_probs(L5)

By default it's set to yellow + entropy, but these can be toggled.  For instance:

compute_probs(L5, H_TL=[0,1], H_terr=[1])