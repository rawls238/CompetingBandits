
# Import BanditAlgorithm classes
from lib.bandit.StaticGreedy import StaticGreedy
from lib.bandit.DynamicEpsilonGreedy import DynamicEpsilonGreedy
from lib.bandit.DynamicGreedy import DynamicGreedy
from lib.bandit.UCB import UCB1WithConstantT
from lib.bandit.ThompsonSampling import ThompsonSampling
from lib.bandit.ExploreThenExploit import ExploreThenExploit
from lib.bandit.MixedGreedy import MixedGreedy
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
T = 1001
NUM_SIMULATIONS = 50

FREE_OBS = False
FREE_OBS_NUM = 200
exp_name = 'fixed_complexity'
print('Exp name', exp_name)
REALIZATIONS_NAME = '' #if you want to pull in past realizations, fill this in with the realizations base name
ERASE_REPUTATION = False
numCores = 10
if len(sys.argv) > 1:
  numCores = sys.argv[1]

AGENT_ALGS = [HardMax]

# valid principal algs are: [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit, ThompsonSampling]
ALG_PAIRS = [(ThompsonSampling, DynamicEpsilonGreedy),(ThompsonSampling, DynamicGreedy), (DynamicGreedy, DynamicEpsilonGreedy)] 
#(ThompsonSampling, ThompsonSampling), (DynamicGreedy, DynamicGreedy), (DynamicEpsilonGreedy, DynamicEpsilonGreedy), 
#(DynamicGreedy, ThompsonSampling), (DynamicEpsilonGreedy, ThompsonSampling), (DynamicEpsilonGreedy, DynamicGreedy)]

def get_needle_in_haystack(starting_mean):
  needle_in_haystack = [bernoulli(starting_mean) for i in xrange(K)]
  needle_in_haystack[int(K/2)] = bernoulli(starting_mean + 0.2)
  return needle_in_haystack

def gen_rand_instance():
  return [bernoulli(np.random.rand()) for i in xrange(np.random.randint(10, 20))]

heavy_tail_prior = beta(0.6, 0.6)

BANDIT_DISTR = {
  'Uniform': None
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
raw_name = exp_base_name + '_raw.csv'
realizations_name = exp_base_name + '_realizations.csv'
dist_name = exp_base_name + '_dist.csv'

INDIVIDUAL_FIELD_NAMES =['Prior', 'N', 'P1 Alg', 'EEOG', 'Instance Complexity', 'Reputation Erased', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret', 'P2 Regret', 'P1 Reputation', 'P2 Reputation', 'Abs Delta Regret']

# fetch distributions from previous run
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

# fetch realizations from previous run
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

### Functions for the experiments drawing from priors ###

# predraw the realizations table so that it is consistent across algorithms.
# TODO: there is a performance gain to be had here by batching the random draws

def get_realizations(K, banditDistrName, banditDistr, startSizes, shouldWrite=True, dist=None, tabl=None, complexityVal=None):
  realDistributions = {}
  realizations = {}
  warmStartRealizations = {}
  if shouldWrite:

    free_obs_dist_writer = csv.writer(dist)
    free_obs_dist_writer.writerow(['Prior'] + [i for i in xrange(K)])

    free_obs_realization_writer = csv.writer(tabl)
    free_obs_realization_writer.writerow(['Prior', 't', 'n'] + [i for i in xrange(K)])

  if REALIZATIONS_NAME and len(REALIZATIONS_NAME) > 0:
    realDistributions = fetch_distributions(REALIZATIONS_NAME, banditDistrName)
    (realizations, warmStartRealizations) = fetch_realizations(REALIZATIONS_NAME, banditDistrName)
  else:
    for q in xrange(NUM_SIMULATIONS):
      realDistributions[q] = getRealDistributionsFromPrior(banditDistrName, banditDistr, K, targetComplexityVal=complexityVal)
      realizations[q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(T)]
      if shouldWrite:
        free_obs_dist_writer.writerow([banditDistrName] + [realDistributions[q][j].mean() for j in xrange(len(realDistributions[q]))])
        free_obs_realization_writer.writerows([[banditDistrName, k, q] + [z for z in realizations[q][k]] for k in xrange(T)])
    for start in startSizes:
      warmStartRealizations[start] = {}
      for q in xrange(NUM_SIMULATIONS):
        if FREE_OBS: # the free obs observations go first, before the warm start observations
          warmStartRealizations[start][q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(FREE_OBS_NUM)]
          warmStartRealizations[start][q] += [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(start)]
          if shouldWrite:
            free_obs_realization_writer.writerows([[banditDistrName, -1*FREE_OBS_NUM - k - 1, q] + [z for z in warmStartRealizations[start][q][k+FREE_OBS_NUM]] for k in xrange(start)])
            free_obs_realization_writer.writerows([[banditDistrName,  -1*k - 1, q] + [z for z in warmStartRealizations[start][q][k]] for k in xrange(FREE_OBS_NUM)])
        else:
          warmStartRealizations[start][q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(start)]
          if shouldWrite:
            free_obs_realization_writer.writerows([[banditDistrName, -1*k-1, q] + [z for z in warmStartRealizations[start][q][k]] for k in xrange(start)])
    
  return (realDistributions, realizations, warmStartRealizations)

# Run a series of experiments that are recorded in CSV files 
def run_experiment(startSizes):
  results = {}
  with open(raw_name, 'w') as raw_csv:
    with open(realizations_name, 'w') as tabl:
      with open(dist_name, 'w') as dist:
        individual_fieldnames = copy(INDIVIDUAL_FIELD_NAMES)
        individual_fieldnames.append('Warm Start')
        individual_writer = csv.DictWriter(raw_csv, fieldnames=individual_fieldnames)
        individual_writer.writeheader()

        for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
          (realDistributions, realizations, warmStartRealizations) = get_realizations(K, banditDistrName, banditDistr, startSizes, shouldWrite=False, dist=dist, tabl=tabl)
          for agentAlg in AGENT_ALGS:
            for (principalAlg1, principalAlg2) in ALG_PAIRS:
              for startSize in startSizes:
                print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with warm start size ' + str(startSize) + ' with prior ' + banditDistrName)
                simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, K=K, T=T, memory=100, warmStartNumObservations=startSize, realizations=realizations[i], warmStartRealizations=warmStartRealizations[startSize][i], freeObsForP2=FREE_OBS, freeObsNum=FREE_OBS_NUM, realDistributions=realDistributions[i], seed=i+1, eraseReputation=ERASE_REPUTATION) for i in xrange(NUM_SIMULATIONS))
                for sim in simResults:
                  for res in sim:
                    t = res['time']
                    regret1 = res['avgRegret1']
                    regret2 = res['avgRegret2']
                    individual_results = {
                      'Warm Start': startSize,
                      'Reputation Erased': ERASE_REPUTATION,
                      'Time Horizon': t,
                      'Prior': banditDistrName,
                      'Agent Alg': agentAlg.__name__,
                      'P1 Alg': principalAlg1.__name__,
                      'P2 Alg': principalAlg2.__name__,
                      'P1 Regret': regret1,
                      'P2 Regret': regret2,
                      'EEOG': res['effectiveEndOfGame'],
                      'Instance Complexity': res['complexity'],
                      'P1 Reputation': res['reputation1'],
                      'P2 Reputation': res['reputation2'],
                      'Abs Delta Regret': np.abs(regret1 - regret2),
                      'Market Share for P1': res['marketShare1'],
                    }
                    individual_writer.writerow(individual_results)





 ### Functions for the complexity experiment ###
def get_distributions(N, banditDistrName, banditDistr, complexityVal=None):
  realDistributions = {}
  for q in xrange(N):
    K = np.random.randint(10, 30)
    realDistributions[q] = getRealDistributionsFromPrior(banditDistrName, banditDistr, K, targetComplexityVal=complexityVal)
  return realDistributions

def get_realizations_with_distr(realDistributions, startSizes, numSim):
  realizations = {}
  warmStartRealizations = {}
  for q in xrange(numSim):
    realizations[q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(T)]
  for start in startSizes:
    warmStartRealizations[start] = {}
    for q in xrange(numSim):
      warmStartRealizations[start][q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(start)]
  return (realizations, warmStartRealizations)


def run_complexity_experiment(startSizes, complexityVals):
  results = {}
  N = 50
  numSim = 50
  with open(raw_name, 'w') as raw_csv:
    individual_fieldnames = copy(INDIVIDUAL_FIELD_NAMES)
    individual_fieldnames.append('Warm Start')
    individual_writer = csv.DictWriter(raw_csv, fieldnames=individual_fieldnames)
    individual_writer.writeheader()
    for complexityVal in complexityVals:
      realDistributions = get_distributions(N, 'FixedComplexity', None, complexityVal=complexityVal)
      for n in xrange(N):
        (realizations, warmStartRealizations) = get_realizations_with_distr(realDistributions, startSizes, numSim)
        print("Complexity val", complexityVal, " Iteration ", n)
        for agentAlg in AGENT_ALGS:
          for (principalAlg1, principalAlg2) in ALG_PAIRS:
            for startSize in startSizes:
              #print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with warm start size ' + str(startSize) + ' with prior Complexity')
              simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, K=K, T=T, memory=100, warmStartNumObservations=startSize, realizations=realizations[i], warmStartRealizations=warmStartRealizations[startSize][i], freeObsForP2=FREE_OBS, freeObsNum=FREE_OBS_NUM, realDistributions=realDistributions[i], seed=i+1, eraseReputation=ERASE_REPUTATION) for i in xrange(numSim))
              for sim in simResults:
                for res in sim:
                  t = res['time']
                  regret1 = res['avgRegret1']
                  regret2 = res['avgRegret2']
                  individual_results = {
                    'Warm Start': startSize,
                    'N': n,
                    'Reputation Erased': ERASE_REPUTATION,
                    'Time Horizon': t,
                    'Prior': 'Complexity',
                    'Agent Alg': agentAlg.__name__,
                    'P1 Alg': principalAlg1.__name__,
                    'P2 Alg': principalAlg2.__name__,
                    'P1 Regret': regret1,
                    'P2 Regret': regret2,
                    'EEOG': res['effectiveEndOfGame'],
                    'Instance Complexity': res['complexity'],
                    'P1 Reputation': res['reputation1'],
                    'P2 Reputation': res['reputation2'],
                    'Abs Delta Regret': np.abs(regret1 - regret2),
                    'Market Share for P1': res['marketShare1'],
                  }
                  individual_writer.writerow(individual_results)

START_SIZES = [50, 100, 200]
COMPLEXITY_VALUES = [100, 500, 1000, 2500, 5000, 7500, 10000]
run_complexity_experiment(START_SIZES, COMPLEXITY_VALUES)
print('all done!')
