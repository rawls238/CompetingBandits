'''The actual bandit instance that stores the true distributions of the arms and draws realizations from the given distributions'''

class BanditProblemInstance:
  def __init__(self, K, T, distributions, realizations = None):
    self.K = K
    self.T = T
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

  def bestArmMean(self):
    return max([distr.mean() for distr in self.distributions])
