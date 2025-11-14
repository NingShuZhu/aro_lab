#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np

from bezier import Bezier
from path import simple_path
from path import computepath
import pybullet as pb
import pinocchio as pin
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

# arguments for integral part (I of PID)
dt = 1/240.0
Ki =200

def controllaw(sim, robot, trajs, tcurrent, cube):
    
    q_des, vq_des, vvq_des = trajs(tcurrent)
    
    # current state
    q, vq = sim.getpybulletstate()
    M = pin.crba(robot.model, robot.data, q)
    h = pin.nle(robot.model, robot.data, q, vq)

    e = q_des - q
    edot = vq_des - vq

    # integral part
    if not hasattr(sim, 'integral_error'):
        sim.integral_error = np.zeros_like(q)

    sim.integral_error += e * dt

    # anti-windup
    sim.integral_error = np.clip(sim.integral_error, -5, 5)

    tau = M @ (vvq_des + Kp * e + Kv * edot + Ki * sim.integral_error) + h
    # inverse dynamics PID control law

    pin.forwardKinematics(robot.model, robot.data, q, vq)
    pin.updateFramePlacements(robot.model, robot.data)
    
    left_hand_joint_id = sim.bullet_names2indices[LEFT_HAND]
    right_hand_joint_id = sim.bullet_names2indices[RIGHT_HAND]
    cube_id = sim.cubeId  

    # get the postitons of the both hands and the cube
    left_pos, _ = pb.getLinkState(sim.robot, left_hand_joint_id)[:2]
    right_pos, _ = pb.getLinkState(sim.robot, right_hand_joint_id)[:2]
    cube_pos, _ = pb.getBasePositionAndOrientation(cube_id)

    left_pos = np.array(left_pos)
    right_pos = np.array(right_pos)
    cube_pos = np.array(cube_pos)

    # calculate the direction of the linear grasping force (from hand to the cube)
    left_force_dir = cube_pos - left_pos
    right_force_dir = cube_pos - right_pos
    left_force_dir /= np.linalg.norm(left_force_dir)
    right_force_dir /= np.linalg.norm(right_force_dir)

    force_magnitude = 50.0  # magnititude of the grasping force
    
    # send the control tau
    sim.step(tau)
    
    # apply the linear grasping force
    pb.applyExternalForce(sim.robot, left_hand_joint_id, (force_magnitude*left_force_dir).tolist(),
                          left_pos.tolist(), pb.WORLD_FRAME)
    pb.applyExternalForce(sim.robot, right_hand_joint_id, (force_magnitude*right_force_dir).tolist(),
                          right_pos.tolist(), pb.WORLD_FRAME)


if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    import time
    
    # robot, sim, cube = setupwithpybullet()
    robot, sim, cube, viz = setupwithpybulletandmeshcat()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    # from path import computepath
    from path2traj import interpolate_path, interpolate_path_t, piecewise_bezier_trajectory, interpolate_path_zero_end_vel, bezier_QP_trajectory_improved
    from config import LEFT_HAND, RIGHT_HAND
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    # a test for holding the cube at a specific position
    # simplepath = simple_path(robot, cube, CUBE_PLACEMENT, q0)
    # simplepath.append(simplepath[-1])
    # simplepath.append(simplepath[-1])
    # simplepath.append(simplepath[-1])
    # simplepath.append(simplepath[-1])
    
    #setting initial configuration
    sim.setqsim(q0)
    
    total_time=4.

    # 5 versions of trajectory modeling, 
    # not recommend bezier ones as they may increase the probability of collision
    # traj = interpolate_path_t(path, total_time)
    # traj = interpolate_path(path, total_time)
    # traj = bezier_QP_trajectory_improved(path, total_time)
    # traj = piecewise_bezier_trajectory(path, total_time)
    traj = interpolate_path_zero_end_vel(path, total_time)
    
    
    tcur = 0.

    # run the simulation
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, traj, tcur, cube)
        # visualize trajectory
        # # q, _, _ = traj(tcur)
        # viz.display(q)
        
        # time.sleep(DT)
        tcur += DT

    time.sleep(1)
    
    
    