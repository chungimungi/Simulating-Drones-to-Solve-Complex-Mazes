import pandas as pd


file_path = 'pathfinding_comparison_results.csv' 
df = pd.read_csv(file_path)

df = df[['Run', 'Pathfinding Time A* (s)', 'Pathfinding Time aStarbin (s)']]

df.columns = ['Run', 'A*', 'aStarbin']

df.set_index('Run', inplace=True)

fastest_each_run = df.idxmin(axis=1)

overall_mean_times = df.mean()

overall_fastest_algorithm = overall_mean_times.idxmin()

print("Fastest algorithm at each run:")
print(fastest_each_run)
print("\nOverall mean Pathfinding Time for each algorithm (in seconds):")
print(overall_mean_times)
print(f"\nOverall fastest algorithm: {overall_fastest_algorithm} by {overall_mean_times.min():.2f} seconds")

