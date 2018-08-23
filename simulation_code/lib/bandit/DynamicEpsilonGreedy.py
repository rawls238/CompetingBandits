from BanditAlgorithm import BanditAlgorithm

import numpy as np

# shouldn't this be a subclass of DynamicGreedy?
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
      self.epsilon = (t + 1)**(-1/3)
    chosenArm = np.argmax([p.mean() for p in self.posteriors])
    if np.random.rand() > self.epsilon:
      return chosenArm
    else:
      return np.random.choice(range(self.banditProblemInstance.K))

