import numpy as np
import matplotlib.pyplot as plt

# A simple 2D GraphSLAM implementation
# Optimize robot trajectory using simulated odometry and landmark measurements.

def simulate_environment():
    # 5-8 landmarks in an L-shaped hospital corridor
    landmarks = np.array([
        [2.0, 2.0],
        [8.0, 2.0],
        [8.0, 8.0],
        [15.0, 8.0],
        [15.0, 2.0],
        [4.0, 5.0]
    ])
    
    # Generate robot true poses: moves along L-shape
    true_poses = []
    x, y, theta = 0.0, 0.0, 0.0
    for i in range(15):
        true_poses.append([x+(i*0.5), y, theta])
    
    x, y, theta = 7.0, 0.0, np.pi/2
    for i in range(10):
        true_poses.append([x, y+(i*0.5), theta])
        
    x, y, theta = 7.0, 4.5, 0.0
    for i in range(15):
        true_poses.append([x+(i*0.5), y, theta])
        
    return landmarks, np.array(true_poses)

def normalize_angle(angle):
    while angle > np.pi: angle -= 2 * np.pi
    while angle < -np.pi: angle += 2 * np.pi
    return angle

def generate_measurements(true_poses, landmarks):
    np.random.seed(42)
    odometry = []
    # Odometry: dx, dy, dtheta
    for i in range(len(true_poses)-1):
        p1 = true_poses[i]
        p2 = true_poses[i+1]
        
        c, s = np.cos(p1[2]), np.sin(p1[2])
        R_inv = np.array([[c, s], [-s, c]])
        
        delta_pos = R_inv @ (p2[:2] - p1[:2])
        dtheta = normalize_angle(p2[2] - p1[2])
        
        # Add noise
        delta_pos += np.random.normal(0, 0.05, 2)
        dtheta += np.random.normal(0, 0.02)
        
        odometry.append([delta_pos[0], delta_pos[1], dtheta])
        
    noisy_poses = [true_poses[0]]
    for i, odo in enumerate(odometry):
        p_prev = noisy_poses[-1]
        c, s = np.cos(p_prev[2]), np.sin(p_prev[2])
        R = np.array([[c, -s], [s, c]])
        
        p_new = np.zeros(3)
        p_new[:2] = p_prev[:2] + R @ odo[:2]
        p_new[2] = normalize_angle(p_prev[2] + odo[2])
        noisy_poses.append(p_new)
        
    observations = []
    # Observations: [pose_idx, landmark_idx, range, bearing]
    for i, pose in enumerate(true_poses):
        obs_at_pose = []
        for j, lm in enumerate(landmarks):
            dx = lm[0] - pose[0]
            dy = lm[1] - pose[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 6.0: # Range limit
                bearing = normalize_angle(np.arctan2(dy, dx) - pose[2])
                dist_noisy = dist + np.random.normal(0, 0.1)
                bearing_noisy = bearing + np.random.normal(0, 0.05)
                obs_at_pose.append([i, j, dist_noisy, bearing_noisy])
        if obs_at_pose:
            observations.extend(obs_at_pose)
            
    return odometry, noisy_poses, observations

def build_and_optimize(noisy_poses, landmarks, odometry, observations):
    # Initialization
    # We will optimize robot poses. For simplicity, assume landmarks are known 
    # (or could be optimized as well, but instructions mention known cylindrical landmarks).
    poses = np.array(noisy_poses)
    
    num_poses = len(poses)
    
    # Information matrix H and vector b
    # We only optimize poses
    info_dim = num_poses * 3
    
    # Covariances (inverse)
    inv_cov_odo = np.diag([1.0/0.05**2, 1.0/0.05**2, 1.0/0.02**2])
    inv_cov_obs = np.diag([1.0/0.1**2, 1.0/0.05**2])
    
    print("Initial error calculation...")
    
    for iteration in range(5):
        H = np.zeros((info_dim, info_dim))
        b = np.zeros((info_dim, 1))
        
        # Odometry constraints
        for i in range(len(odometry)):
            p1 = poses[i]
            p2 = poses[i+1]
            odo = odometry[i]
            
            c, s = np.cos(p1[2]), np.sin(p1[2])
            R_inv = np.array([[c, s], [-s, c]])
            
            pred_delta = R_inv @ (p2[:2] - p1[:2])
            pred_dtheta = normalize_angle(p2[2] - p1[2])
            
            err = np.zeros(3)
            err[:2] = pred_delta - odo[:2]
            err[2] = normalize_angle(pred_dtheta - odo[2])
            
            # Jacobians w.r.t p1 and p2
            A = np.zeros((3, 3))
            B = np.zeros((3, 3))
            
            # Derivative of R_inv * (p2 - p1)
            # R_inv = [c, s; -s, c]
            # p1 = [x1, y1, t1], p2 = [x2, y2, t2]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            A[:2, :2] = -R_inv
            A[0, 2] = -s*dx + c*dy
            A[1, 2] = -c*dx - s*dy
            A[2, 2] = -1
            
            B[:2, :2] = R_inv
            B[2, 2] = 1
            
            idx1 = i*3
            idx2 = (i+1)*3
            
            H[idx1:idx1+3, idx1:idx1+3] += A.T @ inv_cov_odo @ A
            H[idx1:idx1+3, idx2:idx2+3] += A.T @ inv_cov_odo @ B
            H[idx2:idx2+3, idx1:idx1+3] += B.T @ inv_cov_odo @ A
            H[idx2:idx2+3, idx2:idx2+3] += B.T @ inv_cov_odo @ B
            
            b[idx1:idx1+3, 0] += (A.T @ inv_cov_odo @ err)
            b[idx2:idx2+3, 0] += (B.T @ inv_cov_odo @ err)
            
        # Observation constraints
        for obs in observations:
            idx, lm_idx, r, bear = obs
            idx = int(idx)
            lm_idx = int(lm_idx)
            
            p = poses[idx]
            lm = landmarks[lm_idx]
            
            dx = lm[0] - p[0]
            dy = lm[1] - p[1]
            q = dx**2 + dy**2
            dist = np.sqrt(q)
            pred_bear = normalize_angle(np.arctan2(dy, dx) - p[2])
            
            err = np.array([dist - r, normalize_angle(pred_bear - bear)])
            
            # Jacobian w.r.t pose
            J = np.zeros((2, 3))
            J[0, 0] = -dx/dist
            J[0, 1] = -dy/dist
            J[1, 0] = dy/q
            J[1, 1] = -dx/q
            J[1, 2] = -1
            
            pose_idx = idx*3
            H[pose_idx:pose_idx+3, pose_idx:pose_idx+3] += J.T @ inv_cov_obs @ J
            b[pose_idx:pose_idx+3, 0] += (J.T @ inv_cov_obs @ err)
            
        # Fix the first pose to remove gauge freedom
        H[0:3, 0:3] += np.eye(3) * 1e6
        
        # Solve
        delta = np.linalg.solve(H, -b)
        delta = delta.flatten()
        
        # Update poses
        for i in range(num_poses):
            poses[i, 0] += delta[i*3]
            poses[i, 1] += delta[i*3+1]
            poses[i, 2] = normalize_angle(poses[i, 2] + delta[i*3+2])
            
        print(f"Iteration {iteration+1} error norm: {np.linalg.norm(delta)}")
        
    return poses

def main():
    landmarks, true_poses = simulate_environment()
    odometry, noisy_poses, observations = generate_measurements(true_poses, landmarks)
    
    noisy_poses = np.array(noisy_poses)
    optimized_poses = build_and_optimize(noisy_poses, landmarks, odometry, observations)
    
    # Plot true, noisy, and optimized trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(true_poses[:, 0], true_poses[:, 1], 'g-', label='True Trajectory', linewidth=2)
    plt.plot(noisy_poses[:, 0], noisy_poses[:, 1], 'r-', label='Dead Reckoning', linewidth=2)
    plt.plot(optimized_poses[:, 0], optimized_poses[:, 1], 'b-', label='SLAM Optimized', linewidth=2)
    
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='black', marker='*', s=150, label='Landmarks')
    
    plt.title('2D GraphSLAM - Trajectory Correction')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    plt.savefig('outputs/slam_trajectory.png', dpi=150)
    print("Saved plot to outputs/slam_trajectory.png")

if __name__ == '__main__':
    main()
