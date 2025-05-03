import numpy as np

def stanley_steering(x, y, yaw, v, waypoints, last_idx, k=1.0, ks=1e-2, max_steer=np.radians(30)):
    """
    Stanley steering controller.

    Args:
        x, y     : rear axle position of the car
        yaw      : vehicle heading angle (in radians)
        v        : vehicle speed
        waypoints: Nx2 array of path waypoints
        k        : cross-track gain
        ks       : softening term to prevent div by zero
        max_steer: steering angle limits (in radians)

    Returns:
        steer       : steering angle in radians
        target_idx  : index of the nearest waypoint
    """
    # Step 1: Compute front axle position
    L = 2.5  # assume fixed wheelbase
    fx = x + L * np.cos(yaw)
    fy = y + L * np.sin(yaw)

    # Step 2: Find nearest waypoint
    dists = np.sum((waypoints - np.array([fx, fy]))**2, axis=1)
    closest_idx = np.argmin(dists)

    # Step 3: Compute heading of path at that point

    if closest_idx == 0:
        dy = waypoints[closest_idx][1] - waypoints[closest_idx + 1][1]
        dx = waypoints[closest_idx][0] - waypoints[closest_idx + 1][0]
    else:
        dy = waypoints[closest_idx][1] - waypoints[closest_idx - 1][1]
        dx = waypoints[closest_idx][0] - waypoints[closest_idx - 1][0]

    theta_path = np.arctan2(dy, dx)
    # Step 4: Compute heading error

    heading_error = theta_path - yaw

    # Step 5: Compute signed cross-track error

    A = dy/ (dx + 1e-6)  # avoid division by zero
    B = -1
    C = waypoints[closest_idx][1] - (dy/(dx + 1e-6)) * waypoints[closest_idx][0]
    cross_track_error = (A*waypoints[closest_idx][0] + B*waypoints[closest_idx][1] + C) / (np.sqrt(A**2 + B**2))   

    # Step 6: Compute steering using Stanley law
    steer = 1.2*heading_error + np.arctan2((k * 0), (v + ks))   
    steer = np.clip(steer, -max_steer, max_steer)


    print("heading_error: ", heading_error)
    print("cross_track_error: ", cross_track_error)
    print("angle part: ", np.arctan2((k * cross_track_error), (v + ks)))
    target_idx = closest_idx

    return steer, target_idx

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = 0.0
        # you might need to add more variables here... hint: for Integral controller

    def update(self, error, dt):
        '''
        write the update function on your own
        input is error and dt
        output should be the thrust that is provided to drone
        '''
        prop_error = self.Kp * error
        self.integral += error * dt
        integral_error = self.Ki * self.integral
        derivative_error = self.Kd * ((error - self.previous_error)/ dt)
        self.previous_error = error
        thrust = prop_error + integral_error + derivative_error
        return thrust

def pid_throttle():
    pass