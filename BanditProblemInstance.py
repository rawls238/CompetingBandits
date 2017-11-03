class BanditProblemInstance:
  def __init__(self, K, T, distribution='bernoulli'):
    self.K = K
    self.T = T
    self.distribution = distribution
    # come up with K many 0-1 distributions with random means -- these are the K true distributions
    # come up with K many random prior distributions -- this is the starting prior shared by principals and agents

    # new BanditAlgorithm (DynamicGreedy), pass it the prior

  def pullArm(k):
    # draw a realization from distribution k