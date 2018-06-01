
# Import BanditAlgorithm classes
from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.DynamicEpsilonGreedy import DynamicEpsilonGreedy
from lib.bandit.DynamicGreedy import DynamicGreedy
from lib.bandit.UCB import UCB1WithConstantT
from lib.bandit.ThompsonSampling import ThompsonSampling
from lib.bandit.ExploreThenExploit import ExploreThenExploit
from lib.constants import RECORD_STATS_AT, DEFAULT_WARM_START_NUM_OBSERVATIONS
from simulate import simulate, getRealDistributionsFromPrior, initialResultDict


## Import Agent classes
from lib.agent.HardMax import HardMax
from lib.agent.SoftMax import SoftMax
from lib.agent.HardMaxWithRandom import HardMaxWithRandom

# library imports
from joblib import Parallel, delayed
from scipy.stats import bernoulli, beta
from copy import copy, deepcopy
import numpy as np
import csv
import sys
import pickle

K = 10
T = 2002
NUM_SIMULATIONS = 1500

FREE_OBS = False
FREE_OBS_NUM = 200
exp_name = 'vary_warm_start'
REALIZATIONS_NAME = '' #if you want to pull in past realizations, fill this in with the realizations base name
numCores = 10
if len(sys.argv) > 1:
  numCores = sys.argv[1]

AGENT_ALGS = [HardMax]

# valid principal algs are: [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit, ThompsonSampling]
ALG_PAIRS = [(ThompsonSampling, DynamicEpsilonGreedy),(ThompsonSampling, DynamicGreedy), (DynamicGreedy, DynamicEpsilonGreedy)] 
#(ThompsonSampling, ThompsonSampling), (DynamicGreedy, DynamicGreedy), (DynamicEpsilonGreedy, DynamicEpsilonGreedy), 
#(DynamicGreedy, ThompsonSampling), (DynamicEpsilonGreedy, ThompsonSampling), (DynamicEpsilonGreedy, DynamicGreedy)]
default_mean = 0.5
needle_in_haystack = [bernoulli(default_mean) for i in xrange(K)]

needle_in_haystack_50_high = deepcopy(needle_in_haystack)
needle_in_haystack_50_high[int(K/2)] = bernoulli(default_mean + 0.2)

needle_in_haystack_50_medium = deepcopy(needle_in_haystack)
needle_in_haystack_50_medium[int(K/2)] = bernoulli(default_mean + 0.05)

heavy_tail_prior = beta(0.6, 0.6)

BANDIT_DISTR = {
  '.5/.7 Random Draw': None
}

WORKING_DIRECTORY = ''
WORKING_DIRECTORY = '/rigel/home/ga2449/bandits-rl-project/'

if FREE_OBS:
  dir_name = WORKING_DIRECTORY + 'results/free_obs_raw_results/'
  base_name = dir_name + 'free_obs_experiment_'
else:
  dir_name = WORKING_DIRECTORY + 'results/tournament_raw_results/'
  base_name = dir_name + 'tournament_experiment_'

exp_base_name = base_name + exp_name
aggregate_name = exp_base_name + '_aggregate.csv'
raw_name = exp_base_name + '_raw.csv'
realizations_name = exp_base_name + '_realizations.csv'
dist_name = exp_base_name + '_dist.csv'

AGGREGATE_FIELD_NAMES = ['P1 Number of NaNs', 'P2 Number of NaNs', 'Prior', 'P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret Mean', 'P1 Regret Std', 'P2 Regret Mean', 'P2 Regret Std', 'Abs Average Delta Regret']
INDIVIDUAL_FIELD_NAMES =['Prior', 'P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret', 'P2 Regret', 'P1 Reputation', 'P2 Reputation', 'Abs Delta Regret']

def fetch_distributions(filename, priorname):
  realDistributions = {}
  count = 0
  with open(base_name + filename + '_dist.csv', 'rb') as dist_csv:
    dist_reader = csv.reader(dist_csv)
    for row in dist_reader:
      if row[0] != priorname: continue
      realDistributions[count] = [bernoulli(float(row[i])) for i in xrange(1, len(row))]
      count += 1
  return realDistributions

def fetch_realizations(filename, priorname):
  realizations = {}
  warmStartRealizations = {}
  numWarmStartRealizations = DEFAULT_WARM_START_NUM_OBSERVATIONS
  if FREE_OBS:
    numWarmStartRealizations += FREE_OBS_NUM
  with open(base_name + filename + '_realizations.csv', 'rb') as realizations_csv:
    realizations_reader = csv.reader(realizations_csv)
    for row in realizations_reader:
      if row[0] != priorname: continue
      n = int(row[2])
      t = int(row[1])
      if t >= 0: 
        if n not in realizations:
          realizations[n] = [[] for q in xrange(T)]
        realizations[n][t] = [int(row[i]) for i in xrange(3, len(row))]
      else:
        t = -1*t - 1
        if n not in warmStartRealizations:
          warmStartRealizations[n] = [[] for q in xrange(numWarmStartRealizations)]
        warmStartRealizations[n][t] = [int(row[i]) for i in xrange(3, len(row))]
  return (realizations, warmStartRealizations)

def run_experiment(startSizes):
  results = {}
  with open(aggregate_name, 'w') as aggregate_csv:
    with open(raw_name, 'w') as raw_csv:
      with open(realizations_name, 'w') as tabl:
        with open(dist_name, 'w') as dist:
          aggregate_fieldnames = copy(AGGREGATE_FIELD_NAMES)
          aggregate_fieldnames.append('Warm Start')
          aggregate_writer = csv.DictWriter(aggregate_csv, fieldnames=aggregate_fieldnames)
          aggregate_writer.writeheader()

          individual_fieldnames = copy(INDIVIDUAL_FIELD_NAMES)
          individual_fieldnames.append('Warm Start')
          individual_writer = csv.DictWriter(raw_csv, fieldnames=individual_fieldnames)
          individual_writer.writeheader()

          free_obs_dist_writer = csv.writer(dist)
          free_obs_dist_writer.writerow(['Prior'] + [i for i in xrange(K)])

          free_obs_realization_writer = csv.writer(tabl)
          free_obs_realization_writer.writerow(['Prior', 't', 'n'] + [i for i in xrange(K)])

          for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
            realDistributions = {}
            realizations = {}
            warmStartRealizations = {}
            if REALIZATIONS_NAME and len(REALIZATIONS_NAME) > 0:
              realDistributions = fetch_distributions(REALIZATIONS_NAME, banditDistrName)
              (realizations, warmStartRealizations) = fetch_realizations(REALIZATIONS_NAME, banditDistrName)
            else:
              for q in xrange(NUM_SIMULATIONS):
                realDistributions[q] = getRealDistributionsFromPrior(banditDistrName, banditDistr, K)
                free_obs_dist_writer.writerow([banditDistrName] + [realDistributions[q][j].mean() for j in xrange(len(realDistributions[q]))])
                realizations[q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(T)]
                free_obs_realization_writer.writerows([[banditDistrName, k, q] + [z for z in realizations[q][k]] for k in xrange(T)])
              for start in startSizes:
                warmStartRealizations[start] = {}
                for q in xrange(NUM_SIMULATIONS):
                  if FREE_OBS: # the free obs observations go first, before the warm start observations
                    warmStartRealizations[start][q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(FREE_OBS_NUM)]
                    free_obs_realization_writer.writerows([[banditDistrName,  -1*k - 1, q] + [z for z in warmStartRealizations[start][q][k]] for k in xrange(FREE_OBS_NUM)])
                    warmStartRealizations[start][q] += [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(start)]
                    free_obs_realization_writer.writerows([[banditDistrName, -1*FREE_OBS_NUM - k - 1, q] + [z for z in warmStartRealizations[start][q][k+FREE_OBS_NUM]] for k in xrange(start)])
                  else:
                    warmStartRealizations[start][q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(start)]
                    free_obs_realization_writer.writerows([[banditDistrName, -1*k-1, q] + [z for z in warmStartRealizations[start][q][k]] for k in xrange(start)])
              
            for agentAlg in AGENT_ALGS:
              results[agentAlg] = {}
              for (principalAlg1, principalAlg2) in ALG_PAIRS:
                results[agentAlg][(principalAlg1, principalAlg2)] = {}
                for startSize in startSizes:
                  results[agentAlg][(principalAlg1, principalAlg2)][startSize] = {}
                  for t in RECORD_STATS_AT:
                    results[agentAlg][(principalAlg1, principalAlg2)][startSize][t] = deepcopy(initialResultDict)
                  print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with warm start size ' + str(startSize) + ' with prior ' + banditDistrName)
                  simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, K=K, T=T, memory=100, warmStartNumObservations=startSize, realizations=realizations[i], warmStartRealizations=warmStartRealizations[startSize][i], freeObsForP2=FREE_OBS, freeObsNum=FREE_OBS_NUM, realDistributions=realDistributions[i], seed=i+1) for i in xrange(NUM_SIMULATIONS))
                  for sim in simResults:
                    for res in sim:
                      t = res['time']
                      regret1 = res['avgRegret1']
                      regret2 = res['avgRegret2']
                      individual_results = {
                        'Warm Start': startSize,
                        'Time Horizon': t,
                        'Prior': banditDistrName,
                        'Agent Alg': agentAlg.__name__,
                        'P1 Alg': principalAlg1.__name__,
                        'P2 Alg': principalAlg2.__name__,
                        'P1 Regret': regret1,
                        'P2 Regret': regret2,
                        'P1 Reputation': res['reputation1'],
                        'P2 Reputation': res['reputation2'],
                        'Abs Delta Regret': np.abs(regret1 - regret2),
                        'Market Share for P1': res['marketShare1'],
                      }
                      individual_writer.writerow(individual_results)
                      for k, v in res.iteritems():
                        results[agentAlg][(principalAlg1, principalAlg2)][startSize][t][k].append(deepcopy(v))
                  for t in RECORD_STATS_AT:
                    regrets1 = [x for x in results[agentAlg][(principalAlg1, principalAlg2)][startSize][t]['avgRegret1']]
                    regrets2 = [x for x in results[agentAlg][(principalAlg1, principalAlg2)][startSize][t]['avgRegret2']]
                    aggregate_results = {
                      'Warm Start': startSize,
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
                      'Market Share for P1': np.mean(results[agentAlg][(principalAlg1, principalAlg2)][startSize][t]['marketShare1'])
                    }
                    aggregate_writer.writerow(aggregate_results)

  # save "results" to disk, just for convenience, so i can look at them later
  pickle.dump(results, open("bandit_simulations.p", "wb" )) # later, you can load this by doing: results = pickle.load( open("bandit_simulations.p", "rb" ))
  return results

START_SIZES = [1, 5, 10, 15, 20, 50, 100]
run_experiment(START_SIZES)
print('all done!')
#DISCOUNT_FACTORS = [0.5, 0.75, 0.9, 0.99]
#run_discounted_experiment(DISCOUNT_FACTORS)
