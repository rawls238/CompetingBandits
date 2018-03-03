import csv
import numpy as np

AGGREGATE_FIELD_NAMES = ['P1 Number of NaNs', 'P2 Number of NaNs', 'Prior', 'P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret Mean', 'P1 Regret Std', 'P2 Regret Mean', 'P2 Regret Std', 'Abs Average Delta Regret', 'Memory Size']
INDIVIDUAL_FIELD_NAMES =['Prior', 'P1 Alg', 'P2 Alg', 'Time Horizon', 'Agent Alg', 'Market Share for P1', 'P1 Regret', 'P2 Regret', 'Abs Delta Regret']

with open('/Users/garidor/Desktop/bandits-rl-project/results/cur_results/memory_experiment_aggregate_results_2.csv', 'w') as write_csv:
  csvwriter = csv.writer(write_csv)
  csvwriter.writerow(AGGREGATE_FIELD_NAMES)
  results = {}
  with open('/Users/garidor/Desktop/bandits-rl-project/results/cur_results/memory_experiment_raw_results_2.csv', 'rb') as read_csv:
    csvreader = csv.reader(read_csv)
    next(csvreader)
    for row in csvreader:
      key = (row[1], row[2], row[3], row[4], row[9])
      if key not in results:
        results[key] = []
      results[key].append(float(row[5]))
  with open('/Users/garidor/Desktop/bandits-rl-project/results/cur_results/memory_experiment_aggregate_results.csv', 'rb') as read_csv_2:
    csvreader= csv.reader(read_csv_2)
    next(csvreader)
    for row in csvreader:
      key = (row[3], row[4], row[5], row[6], row[13])
      row[7] = np.mean(results[key])
      csvwriter.writerow(row)

