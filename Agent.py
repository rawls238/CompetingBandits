import abc
from InformationSet import InformationSet

class Agent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, principals, priors=None):
    self.principals = principals

    # for each principal store the number of times selected and total reward
    self.informationSet = InformationSet(principals)
    # TODO incorporate agent priors into the initial information set

  @abc.abstractmethod
  def selectPrincipal(self):
    return

  def updateInformationSet(self, reward, principalName):
    self.informationSet.updateInformationSet(reward, principalName)
