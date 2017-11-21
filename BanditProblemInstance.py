'''The actual bandit instance that stores the true distributions of the arms and draws realizations from the given distributions'''

class BanditProblemInstance:
  def __init__(self, K, T, distributions):
    self.K = K
    self.T = T
    self.distributions = distributions

  def pullArm(self, a):
    return self.distributions[a].rvs()
