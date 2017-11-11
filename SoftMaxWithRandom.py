import random as rand
from SoftMax import SoftMax

class SoftMaxWithRandom(SoftMax):
  def __init__(self, principals, priors=None, alpha=0.05, epsilon=0.05):
    self.epsilon = epsilon
    super(SoftMaxWithRandom, self).__init__(principals, priors, alpha)

  def selectPrincipal(self):
    if rand.rand() > self.epsilon:
      return super(HardMaxWithRandom, self).selectPrincipal()
    else:
      principal = self.informationSet.getRandPrincipal()
      return (principal, self.principals[principal])
