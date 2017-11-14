import random as rand
from HardMax import HardMax

class HardMaxWithRandom(HardMax):
  @staticmethod
  def name(self):
    return 'HardMaxWithRandom'

  def __init__(self, principals, priors=None, epsilon=0.05):
    self.epsilon = epsilon
    super(HardMaxWithRandom, self).__init__(principals, priors)

  def selectPrincipal(self):
    if rand.random() > self.epsilon:
      return super(HardMaxWithRandom, self).selectPrincipal()
    else:
      principal = self.informationSet.getRandPrincipal()
      return (principal, self.principals[principal])
