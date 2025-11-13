#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np

from bezier import Bezier
from path import simple_path
import pybullet as pb
import pinocchio as pin
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    # q_of_t, vq_of_t, vvq_of_t = trajs
    q_des, vq_des, vvq_des = trajs(tcurrent)
    # q_des = q_of_t(tcurrent)
    # vq_des = vq_of_t(tcurrent)

    
    # tau = M @ vvq_des + h

    # 当前状态
    q, vq = sim.getpybulletstate()
    M = pin.crba(robot.model, robot.data, q)
    h = pin.nle(robot.model, robot.data, q, vq)

    # PD控制
    # e = q_des - q
    # edot = vq_des - vq
    # inverse dynamics control law
    tau = M @ (vvq_des + Kp * (q_des - q) + Kv * (vq_des - vq)) + h #Kp * e + Kv * edot

    left_hand_joint_id = sim.bullet_names2indices[LEFT_HAND]#'LARM_JOINT5'
    right_hand_joint_id = sim.bullet_names2indices[RIGHT_HAND]#'RARM_JOINT5'
    cube_id = sim.cubeId  # cube 在 pybullet 中的 id

    # 获取左右手和 cube 的位置
    left_pos, _ = pb.getLinkState(sim.robot, left_hand_joint_id)[:2]
    right_pos, _ = pb.getLinkState(sim.robot, right_hand_joint_id)[:2]
    cube_pos, _ = pb.getBasePositionAndOrientation(cube_id)

    left_pos = np.array(left_pos)
    right_pos = np.array(right_pos)
    cube_pos = np.array(cube_pos)

    force_magnitude = 10.0  # 可调抓力大小 [N]

    f_left_dir = cube_pos - left_pos
    f_right_dir = cube_pos - right_pos
    # 归一化方向并设大小
    f_left = (f_left_dir/np.linalg.norm(f_left_dir)) * force_magnitude
    f_right = (f_right_dir/np.linalg.norm(f_right_dir)) * force_magnitude

    left_frame_id = robot.model.getFrameId(LEFT_HAND)   # 确认 LEFT_HAND 是 frame 名
    right_frame_id = robot.model.getFrameId(RIGHT_HAND)

    J_left_6 = pin.computeFrameJacobian(robot.model, robot.data, q, left_frame_id, pin.ReferenceFrame.WORLD)
    J_right_6 = pin.computeFrameJacobian(robot.model, robot.data, q, right_frame_id, pin.ReferenceFrame.WORLD)

    # Jv_left = J_left_6[3:6, :]   # linear velocity jacobian (3 x n)
    # Jv_right = J_right_6[3:6, :]
    Jv_left = J_left_6[:3, :]   # linear velocity jacobian (3 x n)
    Jv_right = J_right_6[:3, :]

    tau_force = Jv_left.T @ f_left + Jv_right.T @ f_right

    tau = np.clip((tau + tau_force), -200, 200)
    # 发送控制量
    sim.step(tau)
    # sim.apply_grasp_force(LEFT_HAND, RIGHT_HAND, force_magnitude=100.0)

    # ------------------ 抓取力 ------------------
    # 取左右手在 pybullet 的 joint index
    # sim.bullet_names2indices
    # bullet_names2indices = {
    #     pb.getJointInfo(sim.robot, i)[1].decode(): i
    #     for i in range(pb.getNumJoints(sim.robot))
    # }
    

    # # 计算抓力方向（指向 cube）
    # left_force_dir = cube_pos - left_pos
    # right_force_dir = cube_pos - right_pos
    # left_force_dir /= np.linalg.norm(left_force_dir)
    # right_force_dir /= np.linalg.norm(right_force_dir)

    

    # # 应用抓力
    # pb.applyExternalForce(sim.robot, left_hand_joint_id, (force_magnitude*left_force_dir).tolist(),
    #                       left_pos.tolist(), pb.WORLD_FRAME)
    # pb.applyExternalForce(sim.robot, right_hand_joint_id, (force_magnitude*right_force_dir).tolist(),
    #                       right_pos.tolist(), pb.WORLD_FRAME)


# def controllaw(sim, robot, trajs, tcurrent, cube):
#     q, vq = sim.getpybulletstate()
#     #TODO 
#     torques = [0.0 for _ in sim.bulletCtrlJointsInPinOrder]
#     sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    import time
    
    # robot, sim, cube = setupwithpybullet()
    robot, sim, cube, viz = setupwithpybulletandmeshcat()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    from path2traj import interpolate_path
    from config import LEFT_HAND, RIGHT_HAND
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    # simplepath = simple_path(robot, cube, CUBE_PLACEMENT, q0)
    
    #setting initial configuration
    sim.setqsim(q0)
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities
    # def maketraj(q0,q1,T): #TODO compute a real trajectory !
    #     q_of_t = Bezier([q0,q0,q1,q1],t_max=T)
    #     vq_of_t = q_of_t.derivative(1)
    #     vvq_of_t = vq_of_t.derivative(1)
    #     return q_of_t, vq_of_t, vvq_of_t
    
    
    #TODO this is just a random trajectory, you need to do this yourself
    total_time=5.
    # trajs = maketraj(q0, qe, total_time)   
    # traj = interpolate_path(simplepath, total_time)
    traj = interpolate_path(path, total_time)

    
    
    tcur = 0.

    # visualize trajectory
    while tcur < total_time:
    #     sim.setqsim(q0)
        rununtil(controllaw, DT, sim, robot, traj, tcur, cube)
        q, _, _ = traj(tcur)
        viz.display(q)
        
        time.sleep(DT)
        tcur += DT

    while tcur < total_time:
    #     sim.setqsim(q0)
        rununtil(controllaw, DT, sim, robot, traj, tcur, cube)
        
        # time.sleep(DT)
        tcur += DT
    
    
    