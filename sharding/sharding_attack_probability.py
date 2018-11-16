import numpy as np
import sympy as sym
import scipy as sp
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


### helper functions ###

# fast binomial function (scipy gives floats for some values)
def binomial(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke.
    See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


# coefficients of generating function corresponding to the sharding problem
def gf_coeffs(n_groups, n_per_group, limit_per_group, total_placed):
    """
    Divide black balls into groups of fixed size, with limits of black balls per group
    n_groups            number of groups
    n_per_group         max balls total per group (of all colors)
    limit_per_group     limit of black balls in a particular group (*strict* inequality)
    total_placed        total number of black balls placed in groups
    """
    
    # get coefficients for a single groups generating function (they are symmetrical)
    # must be passed in descending order
    base_coeffs = []
    for i in range(limit_per_group-1, -1, -1):
        base_coeffs.append(binomial(n_per_group, i))

    # get coefficient for combined generating function: [x^total_placed](g(x))^n_groups
    x = sym.symbols("x")
    gf_base = sym.Poly(base_coeffs, x)
    comb_coeffs = (gf_base**n_groups).coeffs()

    # coefficients are listed in descending order
    # We want coefficient of x^total_placed; list includes x^0
    total_placed_coeff = int(comb_coeffs[-(total_placed+1)])

    return total_placed_coeff


# calculate years to failure given probability of failure
def years_to_failure(prob_failure, rounds_per_year):
    # geometric distribution: number of rounds until failure (i.e. a successful attack)
    # https://en.wikipedia.org/wiki/Geometric_distribution
    return (1/prob_failure)/(rounds_per_year)


### Define input parameters ###

N = 1000        # total number of nodes
p = 0.15        # actual Byzantine percentage
K = int(N*p)    # number of Byzantine nodes
S = 10          # number of shards
n = N//S        # nodes per shard (assumes S evenly divides N)
a = 1/3         # Byzantine fault security limit (alpha)
t = 100000      # number of trials



# N = 1500        # total number of nodes
# p = 0.4         # actual Byzantine percentage
# K = int(N*p)    # number of Byzantine nodes
# S = 10          # number of shards
# n = N//S        # nodes per shard (assumes S evenly divides N)
# a = 1/2         # Byzantine fault security limit (alpha)
# t = 100000      # number of trials







### Methodology 1: Simulation ###

# set up array of K bad nodes and N-K good nodes
nodes = np.array([1]*K + [0]*(N-K))

# Binomial: sampling *with* replacement
# https://en.wikipedia.org/wiki/Binomial_distribution
trials_bn = np.full(t, np.nan)

# Hypergeometric: sampling *without* replacement
# https://en.wikipedia.org/wiki/Hypergeometric_distribution
trials_hg = np.full(t, np.nan)

# run t trials
for i in range(t):
    s_bn = np.random.choice(nodes, size=[S, n], replace=True)
    s_hg = np.random.choice(nodes, size=[S, n], replace=False)
    trials_bn[i] = ((s_bn.sum(axis=1)/n >= a).sum() > 0)
    trials_hg[i] = ((s_hg.sum(axis=1)/n >= a).sum() > 0)

# failure probabilities
pf_bn = trials_bn.sum()/t
pf_hg = trials_hg.sum()/t

# Confidence interval (for proportions)
#https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
# CI = proportion_confint(trials_hg.sum(), t, alpha=0.05, method='normal')  # inaccurate for p near 0 or 1
CI = proportion_confint(trials_hg.sum(), t, alpha=0.05, method='jeffreys')
CI_width = (CI[1] - CI[0])/2




### Methodology 2: Analytical calculation ###

# Correct probability directly from 
n_groups = S
n_per_group = int(N/S)
limit_per_group = int(np.ceil(a*N/S))
total_placed = K
num_success = gf_coeffs(n_groups, n_per_group, limit_per_group, total_placed)
num_total = binomial(N, K)
pf_cf = 1 - num_success / num_total

ytf_hg = years_to_failure(pf_cf, 365)

hg_single = 1 - stats.hypergeom.cdf(int(np.ceil(a*n-1)), N, K, n)


# Binomial direct calculation
bn_single = 1 - sp.stats.binom.cdf(int(np.ceil(a*n-1)), n, K/N)
bn_full = 1 - (1 - bn_single)**S

ytf_bn = years_to_failure(bn_full, 365)








# Print results

print('\nCorrect: Sampling without replacement (hypergeometric)')
print('-----------------------------------------------------')
print('Simulated: {prob}'.format(prob=pf_hg))
print('Analytical: {prob}'.format(prob=pf_cf))
print('Analytical (only first shard): {prob}'.format(prob=hg_single))
print('Years to failure: {ytf}'.format(ytf=ytf_hg))


print('\n Incorrect: Sampling with replacement (binomial)')
print('-----------------------------------------------------')
print('Simulated: {prob}'.format(prob=pf_bn))
print('Analytical: {prob}'.format(prob=bn_full))
print('Analytical (only first shard): {prob}'.format(prob=bn_single))
print('Years to failure: {ytf}'.format(ytf=ytf_bn))

