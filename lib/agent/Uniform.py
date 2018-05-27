from Agent import Agent

# chooses between two principals uniformly at random
class Uniform(Agent):
  def selectPrincipal(self):
    principal = self.informationSet.getRandPrincipal()
    return (principal, self.principals[principal])
