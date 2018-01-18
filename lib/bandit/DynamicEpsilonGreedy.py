from BanditAlgorithm import BanditAlgorithm

import numpy as np
import random as rand

class DynamicEpsilonGreedy(BanditAlgorithm):
  def __init__(self, banditProblemInstance, priors, epsilon=0.05):
    super(DynamicEpsilonGreedy, self).__init__(banditProblemInstance, priors)
    self.epsilon = epsilon

  @staticmethod
  def shorthand():
    return 'DEG'

  def pickAnArm(self):
    chosenArm = np.argmax([p.mean() for p in self.posteriors])
    if rand.random() > self.epsilon:
      return chosenArm
    else:
      return rand.choice(range(self.banditProblemInstance.K))