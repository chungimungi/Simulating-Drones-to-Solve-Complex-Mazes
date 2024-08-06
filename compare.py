import numpy as np
import matplotlib.pyplot as plt
import time
from mazegeneration.algo2d import generate_maze, a_star, heuristic
from mazegeneration.algo2d_bi import bidirectional_a_star

def simulate_uav_2d_comparison(maze, path1, path2, speed_mph):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(maze, cmap='viridis')
    ax1.set_title('A* Algorithm')
    uav_2d_1 = ax1.plot([], [], 'ro', markersize=10)[0]
    
    ax2.imshow(maze, cmap='viridis')
    ax2.set_title('Bidirectional A* Algorithm')
    uav_2d_2 = ax2.plot([], [], 'ro', markersize=10)[0]
    
    plt.ion()
    plt.show()
    
    max_length = max(len(path1), len(path2))
    for i in range(max_length):
        if i < len(path1):
            uav_2d_1.set_data([path1[i][1]], [path1[i][0]])
        if i < len(path2):
            uav_2d_2.set_data([path2[i][1]], [path2[i][0]])
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()

def main():
    maze_size = 500
    maze = generate_maze(maze_size)
    start = (1, 1)
    goal = (maze.shape[0] - 2, maze.shape[1] - 2)
    speed_mph = 45
    
    print("UAV scanning the maze...")
    
    # A* Algorithm
    print("\nUAV calculating optimal path using A*...")
    start_time = time.time()
    path_a_star = a_star(maze, start, goal)
    end_time = time.time()
    pathfinding_time_a_star = end_time - start_time
    
    # Bidirectional A* Algorithm
    print("UAV calculating optimal path using bidirectional A*...")
    start_time = time.time()
    path_bi_a_star = bidirectional_a_star(maze, start, goal)
    end_time = time.time()
    pathfinding_time_bi_a_star = end_time - start_time
    
    if path_a_star and path_bi_a_star:
        # Calculate metrics for A*
        total_distance_a_star = sum(heuristic(path_a_star[i], path_a_star[i + 1]) for i in range(len(path_a_star) - 1))
        speed_mps = speed_mph * 1609.34 / 3600
        grid_size_meters = 2000 / maze_size
        total_time_seconds_a_star = total_distance_a_star * grid_size_meters / speed_mps
        
        # Calculate metrics for Bidirectional A*
        total_distance_bi_a_star = sum(heuristic(path_bi_a_star[i], path_bi_a_star[i + 1]) for i in range(len(path_bi_a_star) - 1))
        total_time_seconds_bi_a_star = total_distance_bi_a_star * grid_size_meters / speed_mps
        
        # Print comparison
        print("\nComparison:")
        print(f"A* Algorithm:")
        print(f"  Path finding time: {pathfinding_time_a_star:.2f} seconds")
        print(f"  Total distance: {total_distance_a_star * grid_size_meters:.2f} meters")
        print(f"  Total traversal time: {total_time_seconds_a_star / 60:.2f} minutes")
        print(f"\nBidirectional A* Algorithm:")
        print(f"  Path finding time: {pathfinding_time_bi_a_star:.2f} seconds")
        print(f"  Total distance: {total_distance_bi_a_star * grid_size_meters:.2f} meters")
        print(f"  Total traversal time: {total_time_seconds_bi_a_star / 60:.2f} minutes")
        
        print("\nSimulating UAV movement...")
        simulate_uav_2d_comparison(maze, path_a_star, path_bi_a_star, speed_mph)
    else:
        print("No path found for one or both algorithms.")

if __name__ == "__main__":
    main()