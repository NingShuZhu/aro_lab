## Software Needed:
Same as the initial `requirements.txt`, ensure the meshcat-server is running on a terminal.

## Introduction of each part:
### inverse_geometry:
**Algorithm used:** numerical inverse kinematics.

**Details:** given an initial configuration $q$, the frame names of the two end-effectors, and the corresponding target poses of the two end-effectors,
iteratively:
1. Calculate and concatenate the error between the current poses and the target poses, calculate and combine the Jacobian metrices of the two ee's.
2. Calculate the optimizing step $\Delta q$ using least square ($\Delta q = \alpha * J^{+} \mathrm{err}$).
3. $$q_{k+1} = q_k \oplus \Delta q$$
4. Stop the iterations when the error is small enough or the maximum number of iterations has been reached.

**Arguments:**
- maximum number of iterations: max_iter=200
- error tolerance: eps=1e-4
- $\alpha$: alpha=0.5



### path:
**Algorithm used:** single-query RRT

**Details:** in the RRT algorithm, given the initial and target cube poses, iteratively:
1. Sample a collision-free random cube pose within the bounded region (with margin on x and y, lift the max z by 0.5) between start and goal, and reject samples in collision.
2. Find the nearest existing node in the tree using a pose-distance metric combining translation and rotation.
3. Linearly interpolate the cube pose to approach the sampled pose from the nearest one, compute the robot configuration for each interpolated pose,
   check whether the step is valid (i.e., the robot configuration can be found and is collision-free (distance to obstacle >= 0.002) and within the joint limits)
5. Add the new node (parent, cube pose, q) to the tree if both the cube pose and robot configuration are feasible.
6. Attempt to connect to the goal from the new node, stop early if the goal pose becomes reachable.

**Arguments:**
- maximun number of nodes in RRT: k=5000
- discretized step distance: discretisationdist=0.005
- maximum step distance: delta=0.05

### trajectory (in `path2traj.py`)
**Algorithm used:** given a path and the total time, simply return a function with the parameter time $t$ that 
given a certain $t$ will return a configuration in the path as well as the corresponding velocity and accelerations desired at that time. 
The starting and final velocities are 0.

QP Bezier and segmented Bezier algorithms are also implemented but the resulting curve will make the probability of collision higher, as it will shrink the path to a lower trajectory, thus closer to the obstacle. 
Even the piecewise Bezier trajectory has the possibility shifting the trajectory to the obstacle in some intermediate segments. 
Therefore, the folded line shape trajectory is used in the control part.

### control:
**Algorithm used:** inverse dynamics + PID, apply the linear grasping force additionally in each time step after setting the calculated tau.

**Details:** 
- inverse dynamics + PID is implemented according to the formula below:

  $\tau = M(q)\left(\ddot{q}_d + K_p e + K_v \dot{e} + K_i \int{e dt} \right) + h(q,\dot{q})$
  * $e = q_d - q$
  * $\dot{e} = \dot{q}_d - \dot{q}$

 - linear force directions are calculated from each hand to the cube (center)

**Arguments:**
- proportional gain: Kp=300
- derivative gain: Kv=2 * np.sqrt(Kp)
- integral gain: Ki=200
- integral dt: dt=1/240.0


## Extra Work
- For inverse grometry part, I implemented and tested other optimization methods: fmin_bfgs and fmin_slsqp.

  In the bfgs method, cost consists of the error between the current end-effectors' poses and the target poses, the penalty of collision, the joint limit costs, and the postral bias.
  Weights and coefficients of each parts were fine-tuned carefully. Scaled random perturbation of the initial configuration was also implemented in case the valid configuration fails to be found (alow 5 attempts).

  SLSQP method utilizes almost the same cost funtion composition, with an additional inequality constraint function to prevent collision.

  However, both of the algorithms are really sensitive to the initial configuration, not so robust and efficient compared to the inverse kinematics methods.

- Developed an algorithm generating a smooth joint trajectory by fitting piecewise Bezier curves between waypoints.
It first initializes each segment as a straight-line Bezier curve.
Then it adjusts the control points to enforce $C^1$ or $C^2$ continuity at the segment boundaries.
It also sets the start and end velocities to zero. The result is a smooth and differentiable joint trajectory.


