using KernelDensity
using CSV
using Query
using DataFrames
using PyPlot

dat = CSV.read("/Users/garidor/Desktop/bandits-rl-project/results/preliminary_raw_results/preliminary_plots_unified.csv")
distribution = "Uniform"

function get_prob_rep_greater(data, alg1, alg2, t)
  alg_1_dat = @from i in data begin
      @where i.Algorithm == alg1 && i.t == t
      @select i
      @collect DataFrame
    end

  alg_2_dat = @from i in data begin
      @where i.Algorithm == alg2 && i.t == t
      @select i
      @collect DataFrame
    end

  rep_diff = alg_1_dat[Symbol("Realized Reputation")] - alg_2_dat[Symbol("Realized Reputation")]
  est = kde(rep_diff)
  return quadgk(x -> pdf(est, x), 0, maximum(est.x))[1]
end


filtered_dat =  @from i in dat begin
  @where i.Distribution == distribution
  @select i
  @collect DataFrame
end

t_vals = collect(range(100,100,20))
probs = []
alg1 = "ThompsonSampling"
alg2 = "DynamicGreedy"
for t in t_vals
  push!(probs, get_prob_rep_greater(filtered_dat, alg1, alg2, t))
end

PyPlot.plot(t_vals, probs)

