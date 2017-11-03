

K = 10
T = 1000
distribution='bernoulli'


# instantiate BanditProblemInstance
banditProblemInstance = BanditProblemInstance(K, T, distribution)

# instantiate 2 principals (who are of some subclass BanditAlgorithm)
principal1 = StaticGreedy(banditProblemInstance)
principal2 = StaticGreedy(banditProblemInstance)

principal1_original = StaticGreedy(banditProblemInstance)
principal2_original = StaticGreedy(banditProblemInstance)

for t in xrange(1,T):
  agent = Agent('HardMax', principal1_original, principal2_original)
  principal = agent.choosePrincipal() #integer, 1 or 2
  # run agent t's response function to choose which principal

  if principal == 1:
    # [k, realization] = principal1.pickAnArm() 
  else:
    principal2.pickAnArm()


