import numpy as np
import matplotlib.pyplot as plt
import random
import time
import heapq

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

# CustomAlgo
class AstarBiasHeuristic:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.open_list = []
        self.came_from = {}
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}
        heapq.heappush(self.open_list, (self.f_score[start], start))
        self.open_set = {start}

    def heuristic(self, a, b):
        # Manhattan distance with a bias towards straight lines
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        return (dx + dy) + (dx - dy) * 0.5  #bias factor

    def get_neighbors(self, node):
        neighbors = []
        x, y = node
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.maze.shape[0] and 0 <= ny < self.maze.shape[1] and self.maze[nx, ny] != 1:
                neighbors.append((nx, ny))
        return neighbors

    def reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_path(self):
        goal = self.goal
        while self.open_list:
            _, current = heapq.heappop(self.open_list)
            if current == goal:
                return self.reconstruct_path(current)
            
            self.open_set.remove(current)
            for neighbor in self.get_neighbors(current):
                tentative_g_score = self.g_score[current] + 1
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in self.open_set:
                        heapq.heappush(self.open_list, (f_score, neighbor))
                        self.open_set.add(neighbor)
                    self.f_score[neighbor] = f_score
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
    maze_size = 50
    maze = generate_maze(maze_size)
    start = (1, 1)
    goal = (maze.shape[0] - 2, maze.shape[1] - 2)

    print("UAV scanning the maze...")

    print("UAV calculating optimal path using A*...")
    start_time = time.time()
    astarBin = AstarBiasHeuristic(maze, start, goal)
    path = astarBin.find_path()
    end_time = time.time()
    pathfinding_time = end_time - start_time

    if path:
        print(f"Path found in {pathfinding_time:.2f} seconds! Simulating UAV movement...")

        total_distance = sum(astarBin.heuristic(path[i], path[i + 1]) for i in range(len(path) - 1))
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
    
    print("UAV calculating optimal path using AstarBiasHeuristic...")
    start_time = time.time()
    AstarBinBin = AstarBinBin(maze, start, goal)
    path = AstarBinBin.find_path()
    end_time = time.time()
    pathfinding_time = end_time - start_time
    
    if path:
        print(f"Path found in {pathfinding_time:.2f} seconds! Simulating UAV movement...")
        
        # Calculate the total distance and time to traverse the path
        total_distance = sum(AstarBinBin.heuristic(path[i], path[i + 1]) for i in range(len(path) - 1))
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