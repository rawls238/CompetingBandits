
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
from collections import Counter
from numba import njit, prange
import numpy as np
import csv
import pickle
import random

K = 3
T = 20002
NUM_SIMULATIONS = 150

FREE_OBS = False
FREE_OBS_NUM = 100
exp_name = 'large_horizon'

AGENT_ALGS = [HardMaxWithRandom, SoftMax]

# valid principal algs are: [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit, ThompsonSampling]
ALG_PAIRS = [(ThompsonSampling, DynamicEpsilonGreedy),(ThompsonSampling, DynamicGreedy), (DynamicGreedy, DynamicEpsilonGreedy)]
            #(DynamicGreedy, ThompsonSampling), (DynamicEpsilonGreedy, ThompsonSampling), (DynamicEpsilonGreedy, DynamicGreedy)]
default_mean = 0.5
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]

needle_in_haystack_50_high = deepcopy(needle_in_haystack)
needle_in_haystack_50_high[int(K/2)] = bernoulli(default_mean + 0.2)

needle_in_haystack_50_medium = deepcopy(needle_in_haystack)
needle_in_haystack_50_medium[int(K/2)] = bernoulli(default_mean + 0.05)

heavy_tail_prior = beta(0.6, 0.6)

BANDIT_DISTR = {
  'Needle In Haystack High': needle_in_haystack_50_high
}

WORKING_DIRECTORY = '/rigel/home/ga2449/bandits-rl-project/'
#WORKING_DIRECTORY = ''

if FREE_OBS:
  dir_name = WORKING_DIRECTORY + 'results/free_obs_raw_results/'
  base_name = dir_name + 'free_obs_experiment_' + exp_name
else:
  dir_name = WORKING_DIRECTORY + 'results/tournament_raw_results/'
  base_name = dir_name + 'tournament_experiment_' + exp_name

aggregate_name = base_name + '_aggregate.csv'
raw_name = base_name + '_raw.csv'
realizations_name = base_name + '_realizations.csv'
dist_name = base_name + '_dist.csv'

AGGREGATE_FIELD_NAMES = ['P1 Number of NaNs', 'P2 Number of NaNs', 'Prior', 'P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret Mean', 'P1 Regret Std', 'P2 Regret Mean', 'P2 Regret Std', 'Abs Average Delta Regret']
INDIVIDUAL_FIELD_NAMES =['Prior', 'P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret', 'P2 Regret', 'Abs Delta Regret']

def run_finite_memory_experiment(memory_sizes):
  results = {}
  with open(aggregate_name, 'w') as aggregate_csv:
    with open(raw_name, 'w') as raw_csv:
      with open(realizations_name, 'w') as tabl:
        with open(dist_name, 'w') as dist:
          aggregate_fieldnames = copy(AGGREGATE_FIELD_NAMES)
          aggregate_fieldnames.append('Memory Size')
          aggregate_writer = csv.DictWriter(aggregate_csv, fieldnames=aggregate_fieldnames)
          aggregate_writer.writeheader()

          individual_fieldnames = copy(INDIVIDUAL_FIELD_NAMES)
          individual_fieldnames.append('Memory Size')
          individual_writer = csv.DictWriter(raw_csv, fieldnames=individual_fieldnames)
          individual_writer.writeheader()

          free_obs_dist_writer = csv.writer(dist)
          free_obs_dist_writer.writerow(['Prior'] + [i for i in xrange(K)])

          free_obs_realization_writer = csv.writer(tabl)
          free_obs_realization_writer.writerow(['Prior', 't', 'n'] + [i for i in xrange(K)])

          for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
            realDistributions = {}
            realizations = {}
            for q in xrange(NUM_SIMULATIONS):
              realDistributions[q] = getRealDistributionsFromPrior(banditDistrName, banditDistr, K)
              free_obs_dist_writer.writerow([banditDistrName] + [realDistributions[q][j].mean() for j in xrange(len(realDistributions[q]))])
              realizations[q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(T)]
              free_obs_realization_writer.writerows([[banditDistrName, k, q] + [z for z in realizations[q][k]] for k in xrange(T)])
            for agentAlg in AGENT_ALGS:
              results[agentAlg] = {}
              for (principalAlg1, principalAlg2) in ALG_PAIRS:
                results[agentAlg][(principalAlg1, principalAlg2)] = {}
                for memory in memory_sizes:
                  results[agentAlg][(principalAlg1, principalAlg2)][memory] = {}
                  for t in RECORD_STATS_AT:
                    results[agentAlg][(principalAlg1, principalAlg2)][memory][t] = deepcopy(initialResultDict)
                  print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with memory ' + str(memory) + ' with prior ' + banditDistrName)
                  #simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, memory=memory, realDistributions=DISTR) for i in xrange(NUM_SIMULATIONS))
                  simResults = run_simulations(principalAlg1, principalAlg2, agentAlg, K, T, memory, realizations, realDistributions)
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
                      'Market Share for P1': np.mean(results[agentAlg][(principalAlg1, principalAlg2)][memory][t]['marketShare1'])
                    }
                    aggregate_writer.writerow(aggregate_results)

  # save "results" to disk, just for convenience, so i can look at them later
  pickle.dump(results, open("bandit_simulations.p", "wb" )) # later, you can load this by doing: results = pickle.load( open("bandit_simulations.p", "rb" ))
  return results

MEMORY_SIZES = [100]
run_finite_memory_experiment(MEMORY_SIZES)
#DISCOUNT_FACTORS = [0.5, 0.75, 0.9, 0.99]
#run_discounted_experiment(DISCOUNT_FACTORS)
