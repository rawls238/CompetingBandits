from StaticGreedy import StaticGreedy
from BanditProblemInstance import BanditProblemInstance


from scipy.stats import bernoulli
from scipy.stats import uniform

K = 2
T = 10
distribution='bernoulli'





# true distributions are:
# arm 1 ~ bernoulli(0.6)   mu_1 = 0.6
# arm 2 ~ bernoulli(0.4)   mu_2 = 0.4
real_distributions = [bernoulli(0.6), bernoulli(0.4)]
#real_distributions = [bernoulli(), bernoulli(P_mean[1].rvs())]

# Bandit algorithms are given priors:
# arm 1 ~ bernoulli(0.5)   mu_1 ~ uniform(0,1)
# arm 2 ~ bernoulli(0.3)   mu_2 ~ uniform(0,0.6)
priors = [bernoulli(0.5), bernoulli(0.3)]

# instantiate BanditProblemInstance
banditProblemInstance = BanditProblemInstance(K, T, real_distributions)



# instantiate 2 principals (who are of some subclass of BanditAlgorithm)
principal1 = StaticGreedy(banditProblemInstance, priors)
principal2 = StaticGreedy(banditProblemInstance, priors)



for t in xrange(1,T):
  principal1.executeStep()
  #agent = Agent('HardMax', principal1, principal2)
  #principal = agent.choosePrincipal() #integer, 1 or 2
  # run agent t's response function to choose which principal

  #if principal == 1:
  #  principal1.pickAnArm() 
  #else:
  #  principal2.pickAnArm()


print(principal1.getArmHistory())




