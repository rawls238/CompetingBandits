from BanditAlgorithm import BanditAlgorithm

import numpy as np


class NonBayesianEpsilonGreedy(BanditAlgorithm):
  def __init__(self, banditProblemInstance, priors, epsilon=0.05, dynamicEpsilon=False):
    super(NonBayesianEpsilonGreedy, self).__init__(banditProblemInstance, priors)
    self.epsilon = epsilon
    self.dynamicEpsilon = dynamicEpsilon

  @staticmethod
  def shorthand():
    return 'NBDEG'

  def pickAnArm(self, t):
    if self.dynamicEpsilon:
      self.epsilon = (t + 1)**(-1/3)
    if np.random.rand() > self.epsilon:
      return np.argmax([mean for mean in self.empiricalMeans])
    else:
      return rand.choice(range(self.banditProblemInstance.K))
