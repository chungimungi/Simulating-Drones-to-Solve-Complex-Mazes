import numpy as np
import heapq

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def get_neighbors(maze, node):
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] != 1:
            neighbors.append((nx, ny))
    return neighbors

def bidirectional_a_star(maze, start, goal):
    forward_heap = [(0, start)]
    backward_heap = [(0, goal)]
    forward_came_from = {}
    backward_came_from = {}
    forward_g_score = {start: 0}
    backward_g_score = {goal: 0}
    forward_f_score = {start: heuristic(start, goal)}
    backward_f_score = {goal: heuristic(goal, start)}
    
    middle = None
    
    while forward_heap and backward_heap:
        # Forward search
        current_forward = heapq.heappop(forward_heap)[1]
        
        if current_forward in backward_came_from:
            middle = current_forward
            break
        
        for neighbor in get_neighbors(maze, current_forward):
            tentative_g_score = forward_g_score[current_forward] + 1
            
            if neighbor not in forward_g_score or tentative_g_score < forward_g_score[neighbor]:
                forward_came_from[neighbor] = current_forward
                forward_g_score[neighbor] = tentative_g_score
                forward_f_score[neighbor] = forward_g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(forward_heap, (forward_f_score[neighbor], neighbor))
        
        # Backward search
        current_backward = heapq.heappop(backward_heap)[1]
        
        if current_backward in forward_came_from:
            middle = current_backward
            break
        
        for neighbor in get_neighbors(maze, current_backward):
            tentative_g_score = backward_g_score[current_backward] + 1
            
            if neighbor not in backward_g_score or tentative_g_score < backward_g_score[neighbor]:
                backward_came_from[neighbor] = current_backward
                backward_g_score[neighbor] = tentative_g_score
                backward_f_score[neighbor] = backward_g_score[neighbor] + heuristic(neighbor, start)
                heapq.heappush(backward_heap, (backward_f_score[neighbor], neighbor))
    
    if middle is None:
        return None
    
    # Reconstruct path
    forward_path = []
    backward_path = []
    
    current = middle
    while current in forward_came_from:
        forward_path.append(current)
        current = forward_came_from[current]
    forward_path.append(start)
    forward_path = forward_path[::-1]
    
    current = middle
    while current in backward_came_from:
        backward_path.append(current)
        current = backward_came_from[current]
    
    return forward_path + backward_path[1:]