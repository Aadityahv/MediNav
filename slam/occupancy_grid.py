import numpy as np
import matplotlib.pyplot as plt

# A 2D occupancy grid mapping algorithm from simulated Lidar scans
# Converts range measurements to log-odds probabilities.

def create_environment():
    # 10m x 10m environment, 0.1m resolution
    grid_size = 100
    res = 0.1
    walls = []
    
    # Outer walls (0, 0) to (10, 10)
    for x in np.arange(0, 10, res):
        walls.append([x, 0])
        walls.append([x, 10])
    for y in np.arange(0, 10, res):
        walls.append([0, y])
        walls.append([10, y])
        
    # L-shape inner walls
    for x in np.arange(2, 6, res):
        walls.append([x, 2])
        walls.append([x, 8])
    for y in np.arange(2.0, 8, res):
        walls.append([6.0, y])
        
    return np.array(walls)

def generate_lidar(pose, walls, max_range=6.0):
    x, y, theta = pose
    ranges = []
    bearings = np.deg2rad(np.arange(-180, 180, 2))
    for ang in bearings:
        # Simple raycasting
        min_dist = max_range
        # For simplicity, calculate distance to nearest wall point along ray
        for w in walls:
            wx, wy = w
            dx = wx - x
            dy = wy - y
            w_ang = np.arctan2(dy, dx)
            # Check if wall point aligns with ray (tolerance)
            import math
            angle_diff = abs(np.arctan2(np.sin(w_ang - (theta + ang)), np.cos(w_ang - (theta + ang))))
            if angle_diff < np.deg2rad(1.5):
                dist = np.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    
        # Add noise
        if min_dist < max_range:
            min_dist += np.random.normal(0, 0.05)
        ranges.append(min_dist)
    return ranges, bearings

def bresenham_line(x_0, y_0, x_1, y_1):
    # Bresenham's line algorithm
    x0, y0 = int(x_0), int(y_0)
    x1, y1 = int(x_1), int(y_1)
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            yield x, y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            yield x, y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    yield x, y

def update_map(l_map, pose, ranges, bearings, max_range=6.0, res=0.1):
    # Log-odds probabilities
    l_occ = 1.0    # p = 0.73
    l_free = -0.5  # p = 0.38
    
    x, y, theta = pose
    gx0, gy0 = x / res, y / res
    
    for r, ang in zip(ranges, bearings):
        # Calculate end point of ray
        gx1, gy1 = (x + r * np.cos(theta + ang)) / res, (y + r * np.sin(theta + ang)) / res
        
        # Ray casting
        cells = list(bresenham_line(gx0, gy0, gx1, gy1))
        
        for cx, cy in cells[:-1]:
            if 0 <= cx < 100 and 0 <= cy < 100:
                l_map[int(cx), int(cy)] += l_free # Free cell
                
        # Update occupied cell if hit
        if r < max_range:
            cx, cy = cells[-1]
            if 0 <= cx < 100 and 0 <= cy < 100:
                l_map[int(cx), int(cy)] += l_occ
                
    return l_map

def main():
    # 10m x 10m grid, 0.1m resolution
    grid_size = 100
    res = 0.1
    
    # Log odds map initialized to 0 (probability 0.5)
    l_map = np.zeros((grid_size, grid_size))
    
    walls = create_environment()
    
    # Generate robot trajectory
    poses = []
    x, y, theta = 1.0, 1.0, 0.0
    for i in range(15):
        poses.append([x+(i*0.5), y, theta])
    
    x, y, theta = 8.5, 1.0, np.pi/2
    for i in range(10):
        poses.append([x, y+(i*0.5), theta])
        
    x, y, theta = 8.5, 6.0, np.pi
    for i in range(15):
        poses.append([x-(i*0.5), y, theta])
        
    # Simulate Lidar and update map
    for p in poses:
        ranges, bearings = generate_lidar(p, walls)
        l_map = update_map(l_map, p, ranges, bearings)
        
    # Convert log-odds to probability
    p_map = 1.0 - (1.0 / (1.0 + np.exp(l_map)))
    
    # Plotting
    plt.figure(figsize=(10, 10))
    # Display the grid with origin at bottom-left
    plt.imshow(p_map.T, cmap='Greys', origin='lower', extent=[0, 10, 0, 10], vmin=0, vmax=1)
    
    # Draw trajectory
    poses = np.array(poses)
    plt.plot(poses[:, 0], poses[:, 1], color='cyan', linestyle='-', linewidth=3, label='Robot Trajectory')
    plt.plot(poses[0, 0], poses[0, 1], 'go', markersize=8, label='Start')
    plt.plot(poses[-1, 0], poses[-1, 1], 'ro', markersize=8, label='End')
    
    # Add an arrow to clear up trajectory direction
    mid_idx = len(poses) // 2
    plt.annotate('', xy=(poses[mid_idx, 0], poses[mid_idx, 1]), 
                 xytext=(poses[mid_idx-1, 0], poses[mid_idx-1, 1]),
                 arrowprops=dict(arrowstyle="->", color='cyan', lw=3, shrinkA=0, shrinkB=0))
    
    plt.title('Occupancy Grid Map')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(False)
    
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    plt.savefig('outputs/occupancy_grid.png', dpi=150)
    np.save("outputs/occupancy_grid.npy", p_map)
    print("Saved plot to outputs/occupancy_grid.png and occupancy_grid.npy")

if __name__ == '__main__':
    main()
