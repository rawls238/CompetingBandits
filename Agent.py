import abc

class Agent:
  __metaclass__ = abc.ABCMeta

  def __init__(self, principals, priors=None):
    self.principals = principals

    # for each principal store the number of times selected and total reward
    self.informationSet = { principal: (0, 0.0) for (principal, v) in self.principals.iteritems() }
    # TODO incorporate agent priors into the initial information set

  @abc.abstractmethod
  def selectPrincipal(self):
    return

  def updateInformationSet(self, reward, principal):
    cur = self.informationSet[principal]
    self.informationSet[principal] = (cur[0] + 1, cur[1] + reward)
