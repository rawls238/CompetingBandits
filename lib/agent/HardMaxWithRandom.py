import numpy as np
from HardMax import HardMax

class HardMaxWithRandom(HardMax):
  # epsilon: the baseline probablility that each principal will be picked with. must be in (0, 0.5).
  # if epsilon=0.1, for example, then 20% of the time, the agent picks randomly between the 2 principals
  def __init__(self, principals, K, priors=None, epsilon=0.05, memory=50):
    super(HardMaxWithRandom, self).__init__(principals, K, priors, memory=memory)
    self.epsilon = epsilon

  def selectPrincipal(self):
    if np.random.rand() > self.epsilon:
      return super(HardMaxWithRandom, self).selectPrincipal()
    else:
      principal = self.informationSet.getRandPrincipal()
      return (principal, self.principals[principal])
