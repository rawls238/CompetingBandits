'''The actual bandit instance that stores the true distributions of the arms and draws realizations from the given distributions'''
import numpy as np

class BanditProblemInstance:
  def __init__(self, K, distributions, realizations = None):
    self.K = K
    self.distributions = distributions
    self.realizations = realizations

  def pullArmWithRandomDraw(self, a):
    return self.distributions[a].rvs()

  def pullArm(self, a, t):
    if self.realizations is None:
      return self.pullArmWithRandomDraw(a)
    return self.realizations[t][a]

  def getMeanOfArm(self, arm):
    return self.distributions[arm].mean()

  def getArmMeans(self):
    return [distr.mean() for distr in self.distributions]

  def getBestArm(self):
    return np.argmax([distr.mean() for distr in self.distributions])

  def bestArmMean(self):
    return max([distr.mean() for distr in self.distributions])

  def getComplexityMetric(self):
    means = [distr.mean() for distr in self.distributions]
    bestArm = max(means)
    metric = 0.0
    for mean in means:
      if mean < bestArm:
        metric += 1.0 / (bestArm - mean)
    return metric

  def setRealizations(self, realizations):
    self.realizations = realizations
