import abc
from InformationSet import InformationSet

class Agent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, principals, priors=None):
    self.principals = principals
    self.priors = priors

    # for each principal store the number of times selected and total reward
    self.informationSet = InformationSet(principals, priors)

  @abc.abstractmethod
  def selectPrincipal(self):
    return

  def updateInformationSet(self, reward, principalName):
    self.informationSet.updateInformationSet(reward, principalName)
