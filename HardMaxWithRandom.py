import random as rand
from HardMax import HardMax

class HardMaxWithRandom(HardMax):
  def __init__(self, principals, K, priors=None, epsilon=0.05):
    super(HardMaxWithRandom, self).__init__(principals, K, priors)
    self.epsilon = epsilon

  def selectPrincipal(self):
    if rand.random() > self.epsilon:
      return super(HardMaxWithRandom, self).selectPrincipal()
    else:
      principal = self.informationSet.getRandPrincipal()
      return (principal, self.principals[principal])
