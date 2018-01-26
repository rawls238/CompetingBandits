from BanditAlgorithm import BanditAlgorithm

import numpy as np
import random as rand

class DynamicEpsilonGreedy(BanditAlgorithm):
  def __init__(self, banditProblemInstance, priors, epsilon=0.05, dynamicEpsilon=False):
    super(DynamicEpsilonGreedy, self).__init__(banditProblemInstance, priors)
    self.epsilon = epsilon
    self.dynamicEpsilon = dynamicEpsilon

  @staticmethod
  def shorthand():
    return 'DEG'

  def pickAnArm(self, t):
    if self.dynamicEpsilon:
      self.epsilon = 1.0 / t
    chosenArm = np.argmax([p.mean() for p in self.posteriors])
    if rand.random() > self.epsilon:
      return chosenArm
    else:
      return rand.choice(range(self.banditProblemInstance.K))
