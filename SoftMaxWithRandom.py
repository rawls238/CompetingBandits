import random as rand
from SoftMax import SoftMax, DEFAULT_ALPHA

class SoftMaxWithRandom(SoftMax):
  def __init__(self, principals, priors=None, alpha=DEFAULT_ALPHA, epsilon=0.05):
    self.epsilon = epsilon
    super(SoftMaxWithRandom, self).__init__(principals, priors, alpha)

  def selectPrincipal(self):
    if rand.random() > self.epsilon:
      return super(SoftMaxWithRandom, self).selectPrincipal()
    else:
      principal = self.informationSet.getRandPrincipal()
      return (principal, self.principals[principal])
