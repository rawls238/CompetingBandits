import unittest

import sys
from joblib import Parallel, delayed

sys.path.append('..')
from lib.bandit.DynamicGreedy import DynamicGreedy
from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.ThompsonSampling import ThompsonSampling
from lib.agent.HardMax import HardMax
from lib.BanditProblemInstance import BanditProblemInstance
from simulate import simulate

from scipy.stats import bernoulli, beta

class TestSimulate(unittest.TestCase):
  def test_simple_simulate(self):
    T = 2
    K = 2
    realizations = [[1, 1], [1, 1]]
    heavy_tail_prior = beta(0.6, 0.6)
    distributions = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]
    N = 10
    total_static = 0
    total_dynamic = 0
    for i in xrange(N):
      res = simulate(DynamicGreedy, DynamicGreedy, HardMax, K, T, realDistributions=distributions, realizations=realizations, warmStartNumObservations=0, recordStatsAt=[1])
      res = simulate(StaticGreedy, StaticGreedy, HardMax, K, T, realDistributions=distributions, realizations=realizations, warmStartNumObservations=0, recordStatsAt=[1])
      total_static += res[0]['marketShare1']
      total_dynamic += res[0]['marketShare1']
    self.assertAlmostEqual(total_static, N/2, delta=N/20)
    self.assertAlmostEqual(total_dynamic, N/2, delta=N/20)

  def test_free_obs(self):
    T = 2
    K = 2
    realizations = [[1, 1], [1, 1]]
    heavy_tail_prior = beta(0.6, 0.6)
    distributions = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]
    N = 10
    total_dynamic = 0
    for i in xrange(N):
      res = simulate(DynamicGreedy, DynamicGreedy, HardMax, K, T, realDistributions=distributions, realizations=realizations, warmStartNumObservations=0, freeObsForP2=True,recordStatsAt=[1])
      total_dynamic += res[0]['marketShare2']
    self.assertEqual(total_dynamic, N)

  def test_parallel(self):
    T = 3
    K = 2
    N = 2
    realizations = {i: [[0, 1] for t in xrange(T)] for i in xrange(N)}
    print(realizations)
    heavy_tail_prior = beta(0.6, 0.6)
    distributions = [bernoulli(heavy_tail_prior.rvs()) for i in xrange(K)]
    res = Parallel(n_jobs=N)(delayed(simulate)(ThompsonSampling, ThompsonSampling, HardMax, K, T, realDistributions=distributions, realizations=realizations[i], warmStartNumObservations=0, recordStatsAt=[100], seed=i+1) for i in xrange(N))
    return True

if __name__ == '__main__':
    unittest.main()
