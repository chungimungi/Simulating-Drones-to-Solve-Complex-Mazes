import pandas as pd
import os


file_path = 'results/pathfinding_comparison_results.csv'
results_folder = 'results'
results_file = os.path.join(results_folder, 'comparison_summary.txt')

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

df = pd.read_csv(file_path)

df = df[['Run', 'Pathfinding Time A* (s)', 'Pathfinding Time AstarBiasHeuristic (s)']]
df.columns = ['Run', 'A*', 'AstarBiasHeuristic']
df.set_index('Run', inplace=True)

fastest_each_run = df.idxmin(axis=1)
overall_mean_times = df.mean()
overall_fastest_algorithm = overall_mean_times.idxmin()

results = [
    "Fastest algorithm at each run:",
    fastest_each_run.to_string(),
    "\nOverall mean Pathfinding Time for each algorithm (in seconds):",
    overall_mean_times.to_string(),
    f"\nOverall fastest algorithm: {overall_fastest_algorithm} by {overall_mean_times.min():.2f} seconds"
]

with open(results_file, 'w') as file:
    file.write("\n".join(results))

print(f"Results saved to {results_file}")
