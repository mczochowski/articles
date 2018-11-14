import numpy as np
import sympy as sym
import scipy as sp

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

# distribute identical balls in distinct containers, with no more than (<=) limit balls per container
def distribute_container_with_limit(balls, containers, limit):
    # https://math.stackexchange.com/questions/98065/how-many-ways-can-b-balls-be-distributed-in-c-containers-with-no-more-than
    # https://math.stackexchange.com/questions/1768917/how-many-ways-are-there-to-distribute-26-identical-balls-into-six-distinct-boxes?rq=1
    res = 0
    for i in range(0, containers+1):
        x = (
            (-1)**i 
            * binomial(containers, i) 
            * binomial(balls + containers - 1  - i * (limit + 1), containers - 1)
        )
        res += x
    return res


N = 1000
p = 0.15
K = 150
S = 10
n = N//S      # nodes per shard
a = 1/3
t = 100000      # number of trials


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


CI_width = 1.96*trials_hg.std()/np.sqrt(t)

# Thinking: 
#   number of successes = number of ways to divide K balls into S bins without exceeding Byz limit
#   number of possibilities = number of ways to divide K balls into S bins without exceeding N/S
#   once malicous nodes are assigned, then non-malicious nodes are deterministic
## pf_cf = 1 - distribute_container_with_limit(K, S, int(a*N/S)-1) / distribute_container_with_limit(K, S, int(N/S)-1)
## 1 - distribute_container_with_limit(K, S, int(a*N/S)-1) / binomial(N,K)

# geometric distribution: number of trials until failure
# https://en.wikipedia.org/wiki/Geometric_distribution
# (1/pf_hg)/(365*2)     # in years to failure



### True distribution direct calculation
# https://math.stackexchange.com/questions/2993130/constrained-combinatorial-question-2-types-of-balls-divided-into-k-groups-with
def gf_coeffs(n_groups, n_per_group, limit_per_group, total_placed):
    base_coeffs = []
    # limit is strict inequality <, not <=
    for i in range(limit_per_group-1, -1, -1):
        base_coeffs.append(binomial(n_per_group, i))

    ### NOTE: numpy polynomial expansion leads to integer overflow for large numbers
    # Wolfram alpha gives correct solution http://www.wolframalpha.com/input/?i=coefficient+of+x%5E25+in+(1%2B10x%2B45x%5E2%2B120x%5E3%2B210x%5E4)%5E10
    # gf_base = np.poly1d(base_coeffs)
    # return (gf_base**n_groups).coeffs[-(total_placed+1)]

    # SymPy
    x = sym.symbols("x")
    gf_base = sym.Poly(base_coeffs, x)
    return int((gf_base**n_groups).coeffs()[-(total_placed+1)])



n_groups = S
n_per_group = int(N/S)
limit_per_group = int(np.ceil(a*N/S))
total_placed = K
num_success = gf_coeffs(n_groups, n_per_group, limit_per_group, total_placed)
num_total = binomial(N, K)
pf_cf = 1 - num_success / num_total

ytf = (1/pf_cf)/(365)

print('Simulated: {prob}'.format(prob=pf_hg))
print('Analytical: {prob}'.format(prob=pf_cf))
print('Years to failure: {ytf}'.format(ytf=ytf))

# Binomial direct calculation
## Worst
bn_single = 1-sp.stats.binom.cdf(int(np.ceil(a*n-1)), n, K/N)
bn_full = 1-(1-bn_single)**10

bn_full/pf_cf


