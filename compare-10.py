import numpy as np
import time
import pandas as pd
from mazegeneration.Normal.algo2d import generate_maze, a_star, heuristic
from mazegeneration.Normal.algo_custom import AstarBiasHeuristic

def run_comparison(num_runs=10):
    results = []
    maze_size = 1000
    speed_mph = 100
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        maze = generate_maze(maze_size)
        start = (1, 1)
        goal = (maze.shape[0] - 2, maze.shape[1] - 2)
        
        print("UAV scanning the maze...")
        
        # A* Algorithm
        print("\nUAV calculating optimal path using A*...")
        start_time = time.time()
        path_a_star = a_star(maze, start, goal)
        end_time = time.time()
        pathfinding_time_a_star = end_time - start_time
        
        # astarbias Algorithm
        print("UAV calculating optimal path using AstarBiasHeuristic...")
        start_time = time.time()
        astarbias = AstarBiasHeuristic(maze, start, goal)
        path_astarbias = astarbias.find_path()
        end_time = time.time()
        pathfinding_time_astarbias = end_time - start_time
        
        if path_a_star and path_astarbias:

            total_distance_a_star = sum(heuristic(path_a_star[i], path_a_star[i + 1]) for i in range(len(path_a_star) - 1))
            speed_mps = speed_mph * 1609.34 / 3600
            grid_size_meters = 1000 / maze_size
            total_time_seconds_a_star = total_distance_a_star * grid_size_meters / speed_mps
            
            total_distance_astarbias = sum(heuristic(path_astarbias[i], path_astarbias[i + 1]) for i in range(len(path_astarbias) - 1))
            total_time_seconds_astarbias = total_distance_astarbias * grid_size_meters / speed_mps
            
            results.append({
                'Run': run + 1,
                'Pathfinding Time A* (s)': pathfinding_time_a_star,
                'Total Distance A* (m)': total_distance_a_star * grid_size_meters,
                'Total Time A* (min)': total_time_seconds_a_star / 60,
                'Pathfinding Time AstarBiasHeuristic (s)': pathfinding_time_astarbias,
                'Total Distance AstarBiasHeuristic (m)': total_distance_astarbias * grid_size_meters,
                'Total Time AstarBiasHeuristic (min)': total_time_seconds_astarbias / 60
            })
        else:
            print("No path found for one or both algorithms.")
    
    df = pd.DataFrame(results)
    df.to_csv('results/pathfinding_comparison_results.csv', index=False)
    print("\nResults saved to 'results/pathfinding_comparison_results.csv'.")

if __name__ == "__main__":
    run_comparison(num_runs=10)