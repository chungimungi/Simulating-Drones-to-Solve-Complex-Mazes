import numpy as np
import time
import pandas as pd
from mazegeneration.Normal.algo2d import generate_maze, a_star, heuristic
from mazegeneration.Normal.algo_custom import AstarBin

def run_comparison(num_runs=10):
    results = []
    maze_size = 500
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
        
        # aStarbin Algorithm
        print("UAV calculating optimal path using aStarbin...")
        start_time = time.time()
        aStarbin = AstarBin(maze, start, goal)
        path_aStarbin = aStarbin.find_path()
        end_time = time.time()
        pathfinding_time_aStarbin = end_time - start_time
        
        if path_a_star and path_aStarbin:
            # Calculate metrics for A*
            total_distance_a_star = sum(heuristic(path_a_star[i], path_a_star[i + 1]) for i in range(len(path_a_star) - 1))
            speed_mps = speed_mph * 1609.34 / 3600
            grid_size_meters = 1000 / maze_size
            total_time_seconds_a_star = total_distance_a_star * grid_size_meters / speed_mps
            
            # Calculate metrics for aStarbin
            total_distance_aStarbin = sum(heuristic(path_aStarbin[i], path_aStarbin[i + 1]) for i in range(len(path_aStarbin) - 1))
            total_time_seconds_aStarbin = total_distance_aStarbin * grid_size_meters / speed_mps
            
            # Store results
            results.append({
                'Run': run + 1,
                'Pathfinding Time A* (s)': pathfinding_time_a_star,
                'Total Distance A* (m)': total_distance_a_star * grid_size_meters,
                'Total Time A* (min)': total_time_seconds_a_star / 60,
                'Pathfinding Time aStarbin (s)': pathfinding_time_aStarbin,
                'Total Distance aStarbin (m)': total_distance_aStarbin * grid_size_meters,
                'Total Time aStarbin (min)': total_time_seconds_aStarbin / 60
            })
        else:
            print("No path found for one or both algorithms.")
    
    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv('pathfinding_comparison_results.csv', index=False)
    print("\nResults saved to 'pathfinding_comparison_results.csv'.")

if __name__ == "__main__":
    run_comparison(num_runs=10)