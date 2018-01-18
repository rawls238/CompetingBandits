import random
from scipy.stats import bernoulli, beta


K = 10
T = 500.0
NUM_SIMULATIONS = 20

DEFAULT_MEMORY = 50
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_ALPHA = 10
DEFAULT_WARM_START_NUM_OBSERVATIONS = 5

DEFAULT_COMMON_PRIOR = [beta(5, 5) for k in xrange(K)]
DEFAULT_PRINCIPAL1PRIORS = [beta(5, 5) for k in xrange(K)]
DEFAULT_PRINCIPAL2PRIORS = DEFAULT_PRINCIPAL1PRIORS

DEFAULT_REAL_DISTRIBUTIONS = [bernoulli(DEFAULT_COMMON_PRIOR[i].rvs()) for i in xrange(K)]

needle_in_haystack_real_distr = [bernoulli(0.5) for i in xrange(K)]
needle_in_haystack_real_distr[int(K/2)] = bernoulli(0.7)

uniform_real_distr = [bernoulli(random.uniform(0.25, 0.75)) for i in xrange(K)]


distributions = {
  'Needle In Haystack - Medium': needle_in_haystack_real_distr,
  'Uniform': uniform_real_distr
}
