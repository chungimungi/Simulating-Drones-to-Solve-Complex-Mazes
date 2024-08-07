import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import time

# Maze generation
def generate_maze(size=500):
    maze = np.ones((size, size), dtype=int)
    stack = [(1, 1)]
    while stack:
        current_cell = stack[-1]
        maze[current_cell] = 0
        neighbors = [
            (current_cell[0] + 2, current_cell[1]),
            (current_cell[0] - 2, current_cell[1]),
            (current_cell[0], current_cell[1] + 2),
            (current_cell[0], current_cell[1] - 2)
        ]
        unvisited = [n for n in neighbors if 0 <= n[0] < size and 0 <= n[1] < size and maze[n] == 1]
        if unvisited:
            next_cell = random.choice(unvisited)
            maze[(current_cell[0] + next_cell[0]) // 2, (current_cell[1] + next_cell[1]) // 2] = 0
            stack.append(next_cell)
        else:
            stack.pop()
    
    maze[1, 1] = 2  
    maze[-2, -2] = 3 
    return maze

# A* pathfinding algorithm
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def get_neighbors(maze, node):
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] != 1:
            neighbors.append((nx, ny))
    return neighbors

def a_star(maze, start, goal):
    heap = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while heap:
        current = heapq.heappop(heap)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in get_neighbors(maze, current):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(heap, (f_score[neighbor], neighbor))
    
    return None

def simulate_uav_2d(maze, path, speed_mph):
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap='viridis')
    ax.set_title('2D Maze')
    uav_2d = ax.plot([], [], 'ro', markersize=10)[0]
    
    plt.ion()
    plt.show()
    
    for step in path:
        uav_2d.set_data([step[1]], [step[0]])
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()

def main():
    maze_size = 500
    maze = generate_maze(maze_size)
    start = (1, 1)
    goal = (maze.shape[0] - 2, maze.shape[1] - 2)
    
    print("UAV scanning the maze...")
    
    print("UAV calculating optimal path...")
    start_time = time.time()
    path = a_star(maze, start, goal)
    end_time = time.time()
    pathfinding_time = end_time - start_time
    
    if path:
        print(f"Path found in {pathfinding_time:.2f} seconds! Simulating UAV movement...")
        
        # Calculate the total distance and time to traverse the path
        total_distance = sum(heuristic(path[i], path[i + 1]) for i in range(len(path) - 1))
        speed_mph = 100
        speed_mps = speed_mph * 1609.34 / 3600  
        grid_size_meters = 1000 / maze_size  
        total_time_seconds = total_distance * grid_size_meters / speed_mps
        
        print(f"Total distance: {total_distance * grid_size_meters:.2f} meters")
        print(f"Total time to traverse path: {total_time_seconds / 60:.2f} minutes")
        
        simulate_uav_2d(maze, path, speed_mph)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
