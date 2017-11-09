class BanditProblemInstance:
  def __init__(self, K, T, distributions):
    self.K = K
    self.T = T
    self.distributions = distributions

  def pullArm(self, a):
    return self.distributions[a-1].rvs()