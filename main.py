
# Import BanditAlgorithm classes
from StaticGreedy import StaticGreedy
from DynamicEpsilonGreedy import DynamicEpsilonGreedy
from DynamicGreedy import DynamicGreedy
from UCB import UCB
from ThompsonSampling import ThompsonSampling
from ExploreThenExploit import ExploreThenExploit
from simulate import simulate, getRealDistributionsFromPrior, initialResultDict
from constants import NUM_SIMULATIONS, K, T, needle_in_haystack_real_distr, uniform_real_distr

## Import Agent classes
from HardMax import HardMax

# library imports
import numpy as np
import csv
from copy import copy, deepcopy
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
import matplotlib.pyplot as plt
import pickle

numCores = multiprocessing.cpu_count()

AGENT_ALGS = [HardMax]

# valid principal algs are: [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit, ThompsonSampling]
PRINCIPAL1_ALGS = [ThompsonSampling]
PRINCIPAL2_ALGS = [ThompsonSampling]
DISTR = uniform_real_distr


AGGREGATE_FIELD_NAMES = ['P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret Mean', 'P1 Regret Std', 'P2 Regret Mean', 'P2 Regret Std', 'Abs Average Delta Regret']
INDIVIDUAL_FIELD_NAMES =['P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret', 'P2 Regret', 'Abs Delta Regret']


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
                res = simulate(principalAlg1, principalAlg2, agentAlg, discountFactor=discount_factor, realDistributions=DISTR)
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
              regrets1 = [reportMean(x) for x in results[agentAlg][principalAlg1][principalAlg2][discount_factor]['avgRegret1']]
              regrets2 = [reportMean(x) for x in results[agentAlg][principalAlg1][principalAlg2][discount_factor]['avgRegret2']]
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
                'Abs Average Delta Regret': np.nanmean([np.abs(regrets1[i] - regrets2[i]) for i in xrange(len(regrets1))]),
                'Market Share for P1': np.mean(results[agentAlg][principalAlg1][principalAlg2][discount_factor]['marketShare1']),
              }
              aggregate_writer.writerow(aggregate_results)

  # save "results" to disk, just for convenience, so i can look at them later
  pickle.dump(results, open("bandit_simulations.p", "wb" )) # later, you can load this by doing: results = pickle.load( open("bandit_simulations.p", "rb" ))
  return results

def run_finite_memory_experiment(memory_sizes):
  results = {}
  with open('results/memory_experiment_aggregate_results.csv', 'w') as aggregate_csv:
    with open('results/memory_experiment_raw_results.csv', 'w') as raw_csv:
      aggregate_fieldnames = copy(AGGREGATE_FIELD_NAMES)
      aggregate_fieldnames.append('Memory Size')
      aggregate_writer = csv.DictWriter(aggregate_csv, fieldnames=aggregate_fieldnames)
      aggregate_writer.writeheader()

      individual_fieldnames = copy(INDIVIDUAL_FIELD_NAMES)
      individual_fieldnames.append('Memory Size')
      individual_writer = csv.DictWriter(raw_csv, fieldnames=individual_fieldnames)
      individual_writer.writeheader()
      for agentAlg in AGENT_ALGS:
        results[agentAlg] = {}
        for principalAlg1 in PRINCIPAL1_ALGS:
          results[agentAlg][principalAlg1] = {}
          for principalAlg2 in PRINCIPAL2_ALGS:
            results[agentAlg][principalAlg1][principalAlg2] = {}
            for memory in memory_sizes:
              results[agentAlg][principalAlg1][principalAlg2][memory] = deepcopy(initialResultDict)
              #realDistributions = {}   # for each experiment do we want a new instance or the same instance across distributions?
              #for i in xrange(NUM_SIMULATIONS):
              #  realDistributions[i] = getRealDistributionsFromPrior(COMMON_PRIOR)
              print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with memory ' + str(memory))
              #simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, memory=memory, realDistributions=DISTR) for i in xrange(NUM_SIMULATIONS))
              simResults = []
              for i in xrange(NUM_SIMULATIONS):
                res = simulate(principalAlg1, principalAlg2, agentAlg, discountFactor=discount_factor, realDistributions=DISTR)
                simResults.append(res)
              for res in simResults:
                regret1 = res['avgRegret1'] or 0.0
                regret2 = res['avgRegret2'] or 0.0 #fix this
                individual_results = {
                  'Memory Size': memory,
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
                  results[agentAlg][principalAlg1][principalAlg2][memory][k].append(deepcopy(v))

              def reportMean(x):
                if x is None:
                  return np.nan
                return x
              regrets1 = [reportMean(x) for x in results[agentAlg][principalAlg1][principalAlg2][memory]['avgRegret1']]
              regrets2 = [reportMean(x) for x in results[agentAlg][principalAlg1][principalAlg2][memory]['avgRegret2']]
              aggregate_results = {
                'Memory Size': memory,
                'Time Horizon': T,
                'Agent Alg': agentAlg.__name__,
                'P1 Alg': principalAlg1.__name__,
                'P2 Alg': principalAlg2.__name__,
                'P1 Regret Mean': np.nanmean(regrets1),
                'P2 Regret Mean': np.nanmean(regrets2),
                'P1 Regret Std': np.nanstd(regrets1),
                'P2 Regret Std': np.nanstd(regrets2),
                'Abs Average Delta Regret': np.nanmean([np.abs(regrets1[i] - regrets2[i]) for i in xrange(len(regrets1))]),
                'Market Share for P1': np.mean(results[agentAlg][principalAlg1][principalAlg2][memory]['marketShare1']),
              }
              print(aggregate_results)
              aggregate_writer.writerow(aggregate_results)

  # save "results" to disk, just for convenience, so i can look at them later
  pickle.dump(results, open("bandit_simulations.p", "wb" )) # later, you can load this by doing: results = pickle.load( open("bandit_simulations.p", "rb" ))
  return results

MEMORY_SIZES = [1, 5, 100, 500]
run_finite_memory_experiment(MEMORY_SIZES)
DISCOUNT_FACTORS = [0.5, 0.75, 0.9, 0.99]
#run_discounted_experiment(DISCOUNT_FACTORS)
