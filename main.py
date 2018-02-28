
# Import BanditAlgorithm classes
from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.DynamicEpsilonGreedy import DynamicEpsilonGreedy
from lib.bandit.DynamicGreedy import DynamicGreedy
from lib.bandit.UCB import UCB1WithConstantT
from lib.bandit.ThompsonSampling import ThompsonSampling
from lib.bandit.ExploreThenExploit import ExploreThenExploit
from lib.constants import RECORD_STATS_AT
from simulate import simulate, getRealDistributionsFromPrior, initialResultDict

## Import Agent classes
from lib.agent.HardMax import HardMax
from lib.agent.SoftMax import SoftMax
from lib.agent.HardMaxWithRandom import HardMaxWithRandom

# library imports
from scipy.stats import bernoulli, beta
from copy import copy, deepcopy
from joblib import Parallel, delayed
from collections import Counter
import numpy as np
import csv
import multiprocessing
import pickle
import random

numCores = multiprocessing.cpu_count()

K = 3
T = 5002
NUM_SIMULATIONS = 100

AGENT_ALGS = [HardMax, HardMaxWithRandom]

# valid principal algs are: [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit, ThompsonSampling]
ALG_PAIRS = [(ThompsonSampling, DynamicEpsilonGreedy),(ThompsonSampling, DynamicGreedy), (DynamicGreedy, DynamicEpsilonGreedy)]

default_mean = 0.5
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]

needle_in_haystack_50_high = deepcopy(needle_in_haystack)
needle_in_haystack_50_high[int(K/2)] = bernoulli(default_mean + 0.2)

needle_in_haystack_50_medium = deepcopy(needle_in_haystack)
needle_in_haystack_50_medium[int(K/2)] = bernoulli(default_mean + 0.05)

heavy_tail_prior = beta(0.6, 0.6)

BANDIT_DISTR = {
  'Uniform': None,
  'Heavy Tail' :heavy_tail_prior,
  'Needle In Haystack High': needle_in_haystack_50_high,
  'Needle In Haystack Medium': needle_in_haystack_50_medium,
}



AGGREGATE_FIELD_NAMES = ['P1 Number of NaNs', 'P2 Number of NaNs', 'Prior', 'P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret Mean', 'P1 Regret Std', 'P2 Regret Mean', 'P2 Regret Std', 'Abs Average Delta Regret']
INDIVIDUAL_FIELD_NAMES =['Prior', 'P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret', 'P2 Regret', 'Abs Delta Regret']


def run_discounted_experiment(discount_factors):
  results = {}
  with open('results/discounted_experiment_aggregate_results.csv', 'w') as aggregate_csv:
    with open('results/discounted_experiment_raw_results.csv', 'w') as raw_csv:
      aggregate_fieldnames = copy(AGGREGATE_FIELD_NAMES)
      aggregate_fieldnames.append('Discount Factor')
      aggregate_writer = csv.DictWriter(aggregate_csv, fieldnames=aggregate_fieldnames)
      aggregate_writer.writeheader()

      individual_fieldnames = copy(INDIVIDUAL_FIELD_NAMES)
      individual_fieldnames.append('Discount Factor')
      individual_writer = csv.DictWriter(raw_csv, fieldnames=individual_fieldnames)
      individual_writer.writeheader()

      for agentAlg in AGENT_ALGS:
        results[agentAlg] = {}
        for principalAlg1 in PRINCIPAL1_ALGS:
          results[agentAlg][principalAlg1] = {}
          for principalAlg2 in PRINCIPAL2_ALGS:
            results[agentAlg][principalAlg1][principalAlg2] = {}
            for discount_factor in discount_factors:
              results[agentAlg][principalAlg1][principalAlg2][discount_factor] = deepcopy(initialResultDict)
              #realDistributions = {}   # for each experiment do we want a new instance or the same instance across distributions?
              #for i in xrange(NUM_SIMULATIONS):
              #  realDistributions[i] = getRealDistributionsFromPrior(COMMON_PRIOR)
              print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with discount factor ' + str(discount_factor))
              #simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, discountFactor=discount_factor, realDistributions=DISTR) for i in xrange(NUM_SIMULATIONS))
              simResults = []
              for i in xrange(NUM_SIMULATIONS):
                res = simulate(principalAlg1, principalAlg2, agentAlg, K=K, T=T, discountFactor=discount_factor, realDistributions=DISTR)
                simResults.append(res)

              for res in simResults:
                regret1 = res['avgRegret1'] or 0.0
                regret2 = res['avgRegret2'] or 0.0 #fix this
                individual_results = {
                  'Discount Factor': discount_factor,
                  'Time Horizon': T,
                  'Agent Alg': agentAlg.__name__,
                  'P1 Alg': principalAlg1.__name__,
                  'P2 Alg': principalAlg2.__name__,
                  'P1 Regret': regret1,
                  'P2 Regret': regret2,
                  'Abs Delta Regret': np.abs(regret1 - regret2),
                  'Market Share for P1': res['marketShare1'],
                }
                individual_writer.writerow(individual_results)
                for k, v in res.iteritems():
                  results[agentAlg][principalAlg1][principalAlg2][discount_factor][k].append(deepcopy(v))

              def reportMean(x):
                if x is None:
                  return np.nan
                return x
              regrets1 = [reportMean(x) for x in results[agentAlg][(principalAlg1, principalAlg2)][discount_factor]['avgRegret1']]
              regrets2 = [reportMean(x) for x in results[agentAlg][(principalAlg1, principalAlg2)][discount_factor]['avgRegret2']]
              aggregate_results = {
                'Discount Factor': discount_factor,
                'Time Horizon': T,
                'Agent Alg': agentAlg.__name__,
                'P1 Alg': principalAlg1.__name__,
                'P2 Alg': principalAlg2.__name__,
                'P1 Regret Mean': np.nanmean(regrets1),
                'P2 Regret Mean': np.nanmean(regrets2),
                'P1 Regret Std': np.nanstd(regrets1),
                'P2 Regret Std': np.nanstd(regrets2),
                'Abs Average Delta Regret': np.mean([np.abs(regrets1[i] - regrets2[i]) for i in xrange(len(regrets1))]),
                'Market Share for P1': np.mean(results[agentAlg][(principalAlg1, principalAlg2)][discount_factor]['marketShare1']),
              }
              aggregate_writer.writerow(aggregate_results)

  # save "results" to disk, just for convenience, so i can look at them later
  pickle.dump(results, open("bandit_simulations.p", "wb" )) # later, you can load this by doing: results = pickle.load( open("bandit_simulations.p", "rb" ))
  return results

def run_finite_memory_experiment(memory_sizes):
  results = {}
  with open('results/free_obs_experiment_aggregate_results_3.csv', 'w') as aggregate_csv:
    with open('results/free_obs_experiment_raw_results_3.csv', 'w') as raw_csv:
      aggregate_fieldnames = copy(AGGREGATE_FIELD_NAMES)
      aggregate_fieldnames.append('Memory Size')
      aggregate_writer = csv.DictWriter(aggregate_csv, fieldnames=aggregate_fieldnames)
      aggregate_writer.writeheader()

      individual_fieldnames = copy(INDIVIDUAL_FIELD_NAMES)
      individual_fieldnames.append('Memory Size')
      individual_writer = csv.DictWriter(raw_csv, fieldnames=individual_fieldnames)
      individual_writer.writeheader()

      for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
        for agentAlg in AGENT_ALGS:
          results[agentAlg] = {}
          for (principalAlg1, principalAlg2) in ALG_PAIRS:
            results[agentAlg][(principalAlg1, principalAlg2)] = {}
            for memory in memory_sizes:
              results[agentAlg][(principalAlg1, principalAlg2)][memory] = {}
              for t in RECORD_STATS_AT:
                results[agentAlg][(principalAlg1, principalAlg2)][memory][t] = deepcopy(initialResultDict)
              realDistributions = {}
              for i in xrange(NUM_SIMULATIONS):
                realDistributions[i] = getRealDistributionsFromPrior(banditDistrName, banditDistr, K)
              print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with memory ' + str(memory) + ' with prior ' + banditDistrName)
              #simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, memory=memory, realDistributions=DISTR) for i in xrange(NUM_SIMULATIONS))
              simResults = []
              for i in xrange(NUM_SIMULATIONS):
                res = simulate(principalAlg1, principalAlg2, agentAlg, K=K, T=T, memory=memory, realDistributions=realDistributions[i])
                simResults.append(res)
              for sim in simResults:
                for res in sim:
                  t = res['time']
                  regret1 = res['avgRegret1']
                  regret2 = res['avgRegret2']
                  individual_results = {
                    'Memory Size': memory,
                    'Time Horizon': t,
                    'Prior': banditDistrName,
                    'Agent Alg': agentAlg.__name__,
                    'P1 Alg': principalAlg1.__name__,
                    'P2 Alg': principalAlg2.__name__,
                    'P1 Regret': regret1,
                    'P2 Regret': regret2,
                    'Abs Delta Regret': np.abs(regret1 - regret2),
                    'Market Share for P1': res['marketShare1'],
                  }
                  individual_writer.writerow(individual_results)
                  for k, v in res.iteritems():
                    results[agentAlg][(principalAlg1, principalAlg2)][memory][t][k].append(deepcopy(v))
              for t in RECORD_STATS_AT:
                regrets1 = [x for x in results[agentAlg][(principalAlg1, principalAlg2)][memory][t]['avgRegret1']]
                regrets2 = [x for x in results[agentAlg][(principalAlg1, principalAlg2)][memory][t]['avgRegret2']]
                aggregate_results = {
                  'Memory Size': memory,
                  'Time Horizon': t,
                  'Agent Alg': agentAlg.__name__,
                  'P1 Alg': principalAlg1.__name__,
                  'P2 Alg': principalAlg2.__name__,
                  'P1 Regret Mean': np.nanmean(regrets1),
                  'P1 Number of NaNs': regrets1.count(np.nan),
                  'P2 Regret Mean': np.nanmean(regrets2),
                  'P2 Number of NaNs': regrets2.count(np.nan),
                  'P1 Regret Std': np.nanstd(regrets1),
                  'P2 Regret Std': np.nanstd(regrets2),
                  'Prior': banditDistrName,
                  'Abs Average Delta Regret': np.nanmean([np.abs(regrets1[i] - regrets2[i]) for i in xrange(len(regrets1))]),
                  'Market Share for P1': np.mean(results[agentAlg][(principalAlg1, principalAlg2)][memory][t]['marketShare1']),
                }
                aggregate_writer.writerow(aggregate_results)

  # save "results" to disk, just for convenience, so i can look at them later
  pickle.dump(results, open("bandit_simulations.p", "wb" )) # later, you can load this by doing: results = pickle.load( open("bandit_simulations.p", "rb" ))
  return results

MEMORY_SIZES = [100]
run_finite_memory_experiment(MEMORY_SIZES)
#DISCOUNT_FACTORS = [0.5, 0.75, 0.9, 0.99]
#run_discounted_experiment(DISCOUNT_FACTORS)
