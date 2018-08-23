
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
T = 2001
NUM_SIMULATIONS = 1000

FREE_OBS = True
ERASE_REP = True
ERASE_INFO = False
FREE_OBS_NUM = 200
exp_name = 'full_sim_no_rep'
print('Exp name', exp_name)
REALIZATIONS_NAME = 'preliminary' #if you want to pull in past realizations, fill this in with the realizations base name
numCores = 12
if len(sys.argv) > 1:
  numCores = sys.argv[1]

AGENT_ALGS = [HardMax]

# valid principal algs are: [StaticGreedy, UCB, DynamicEpsilonGreedy, DynamicGreedy, ExploreThenExploit, ThompsonSampling]
ALG_PAIRS = [(ThompsonSampling, DynamicEpsilonGreedy),(ThompsonSampling, DynamicGreedy), (DynamicGreedy, DynamicEpsilonGreedy),
(ThompsonSampling, ThompsonSampling), (DynamicGreedy, DynamicGreedy), (DynamicEpsilonGreedy, DynamicEpsilonGreedy),
(DynamicGreedy, ThompsonSampling), (DynamicEpsilonGreedy, ThompsonSampling), (DynamicEpsilonGreedy, DynamicGreedy)]

def get_needle_in_haystack(starting_mean):
  needle_in_haystack = [bernoulli(starting_mean) for i in xrange(K)]
  needle_in_haystack[int(K/2)] = bernoulli(starting_mean + 0.2)
  return needle_in_haystack

def gen_rand_instance():
  return [bernoulli(np.random.rand()) for i in xrange(np.random.randint(10, 20))]

heavy_tail_prior = beta(0.6, 0.6)

BANDIT_DISTR = {
  'Heavy Tail': heavy_tail_prior,
  'Uniform': None,
  'Needle In Haystack': get_needle_in_haystack(0.5),
  '.5/.7 Random Draw': None
}

WORKING_DIRECTORY = ''
#WORKING_DIRECTORY = '/rigel/home/ga2449/bandits-rl-project/'

instance_dir = WORKING_DIRECTORY + 'results/preliminary_raw_results/'
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

INDIVIDUAL_FIELD_NAMES =['Prior', 'Erase Info', 'Erase Reputation', 'N', 'P1 Alg', 'EEOG', 'Instance Complexity', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret', 'P2 Regret', 'P1 Reputation', 'P2 Reputation', 'Abs Delta Regret']

# fetch distributions from previous run
def fetch_distributions(filename, priorname):
  realDistributions = {}
  count = 0
  with open(instance_dir + filename + '_dist.csv', 'rb') as dist_csv:
    dist_reader = csv.reader(dist_csv)
    for row in dist_reader:
      if row[0] != priorname: continue
      realDistributions[count] = [bernoulli(float(row[i])) for i in xrange(1, len(row))]
      count += 1
  return realDistributions

# fetch realizations from previous run
def fetch_realizations(filename, priorname, maxWarmStart, freeObsNum=FREE_OBS_NUM):
  realizations = {}
  numObs = maxWarmStart + freeObsNum + T
  with open(instance_dir + filename + '_realizations.csv', 'rb') as realizations_csv:
    realizations_reader = csv.reader(realizations_csv)
    for row in realizations_reader:
      if row[0] != priorname: continue
      n = int(row[2])
      t = int(row[1])
      if n not in realizations:
        realizations[n] = [[] for q in xrange(numObs)]
      realizations[n][t] = [int(row[i]) for i in xrange(3, len(row))]

  return realizations

### Functions for the experiments drawing from priors ###

# predraw the realizations table so that it is consistent across algorithms.
# TODO: there is a performance gain to be had here by batching the random draws

def get_realizations(K, banditDistrName, banditDistr, startSizes, shouldWrite=True, dist=None, tabl=None, complexityVal=None):
  realDistributions = {}
  realizations = {}
  maxWarmStart = 200
  obsToGen = T+maxWarmStart+FREE_OBS_NUM
  if shouldWrite:

    free_obs_dist_writer = csv.writer(dist)
    free_obs_dist_writer.writerow(['Prior'] + [i for i in xrange(K)])

    free_obs_realization_writer = csv.writer(tabl)
    free_obs_realization_writer.writerow(['Prior', 't', 'n'] + [i for i in xrange(K)])

  if REALIZATIONS_NAME and len(REALIZATIONS_NAME) > 0:
    realDistributions = fetch_distributions(REALIZATIONS_NAME, banditDistrName)
    realizations = fetch_realizations(REALIZATIONS_NAME, banditDistrName, maxWarmStart)
  else:
    for q in xrange(NUM_SIMULATIONS):
      realDistributions[q] = getRealDistributionsFromPrior(banditDistrName, banditDistr, K, targetComplexityVal=complexityVal)
      realizations[q] = [[realDistributions[q][j].rvs() for j in xrange(len(realDistributions[q]))] for k in xrange(obsToGen)]
      if shouldWrite:
        free_obs_dist_writer.writerow([banditDistrName] + [realDistributions[q][j].mean() for j in xrange(len(realDistributions[q]))])
        free_obs_realization_writer.writerows([[banditDistrName, k, q] + [z for z in realizations[q][k]] for k in xrange(obsToGen)])
  return (realDistributions, realizations)

# Run a series of experiments that are recorded in CSV files 
def run_experiment(startSizes):
  results = {}
  maxStart = 200
  with open(raw_name, 'w') as raw_csv:
    individual_fieldnames = copy(INDIVIDUAL_FIELD_NAMES)
    individual_fieldnames.append('Warm Start')
    individual_writer = csv.DictWriter(raw_csv, fieldnames=individual_fieldnames)
    individual_writer.writeheader()

    for (banditDistrName, banditDistr) in BANDIT_DISTR.iteritems():
      (realDistributions, realizations) = get_realizations(K, banditDistrName, banditDistr, startSizes, shouldWrite=False)
      for agentAlg in AGENT_ALGS:
        for (principalAlg1, principalAlg2) in ALG_PAIRS:
          for startSize in startSizes:
            print('Running ' + agentAlg.__name__ + ' and principal 1 playing ' + principalAlg1.__name__ + ' and principal 2 playing ' + principalAlg2.__name__ + ' with warm start size ' + str(startSize) + ' with prior ' + banditDistrName)
            simResults = Parallel(n_jobs=numCores)(delayed(simulate)(principalAlg1, principalAlg2, agentAlg, maxStart, K=K, T=T, warmStartNumObservations=startSize, realizations=realizations[i], eraseReputation=ERASE_REP, eraseInformation=ERASE_INFO, freeObsForP2=FREE_OBS, freeObsNum=FREE_OBS_NUM, realDistributions=realDistributions[i], seed=i+1) for i in xrange(NUM_SIMULATIONS))
            for i in xrange(len(simResults)):
              sim = simResults[i]
              for res in sim:
                t = res['time']
                regret1 = res['avgRegret1']
                regret2 = res['avgRegret2']
                individual_results = {
                  'Warm Start': startSize,
                  'Time Horizon': t,
                  'Prior': banditDistrName,
                  'N': i,
                  'Erase Reputation': ERASE_REP,
                  'Erase Info': ERASE_INFO,
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


START_SIZES = [20]
#COMPLEXITY_VALUES = [100, 500, 1000, 2500, 5000, 7500, 10000]
#run_complexity_experiment(START_SIZES, COMPLEXITY_VALUES)
run_experiment(START_SIZES)
print('all done!')
