#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement
from scipy.optimize import fmin_bfgs,fmin_slsqp
import meshcat

# def cost(frame_l_id=None, frame_r_id=None, target_pos_l=None, target_pos_r=None):
#     pose_l = robot.data.oMf[frame_l_id]
#     pose_r = robot.data.oMf[frame_r_id]
#     error = 0.5 * (norm(pose_l.translation - target_pos_l)**2 + norm(pose_r.translation - target_pos_r)**2)
#     return error

def cost(frame_l_id=None, frame_r_id=None, target_tf_l=None, target_tf_r=None):
    # target_tf_* are pin.SE3 (or objects with .translation and .rotation)
    pose_l = robot.data.oMf[frame_l_id]
    pose_r = robot.data.oMf[frame_r_id]

    # position error
    e_pos_l = pose_l.translation - target_tf_l.translation
    e_pos_r = pose_r.translation - target_tf_r.translation

    # orientation error: relative rotation from current -> target, then log map to so(3)
    R_err_l = target_tf_l.rotation.dot(pose_l.rotation.T)
    R_err_r = target_tf_r.rotation.dot(pose_r.rotation.T)
    e_ori_l = pin.log3(R_err_l)   # so(3) vector: axis * angle
    e_ori_r = pin.log3(R_err_r)

    # weights
    w_pos = 1.0
    w_ori = 5.0

    error = 0.5 * ( w_pos*(norm(e_pos_l)**2 + norm(e_pos_r)**2) + w_ori*(norm(e_ori_l)**2 + norm(e_ori_r)**2) )
    return error

def constraint(q):
    if collision(robot, q):
        res = 1.0
    else:
        res = 0
    return res

def penalty(q, frame_l_id=None, frame_r_id=None, target_tf_l=None, target_tf_r=None):
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    return cost(frame_l_id, frame_r_id, target_tf_l, target_tf_r) + 10 * constraint(q)

def callback_viz(q):
    updatevisuals(viz, robot, cube, q)

def bfgs(penalty_func, q, frame_l_id, frame_r_id, target_pos_l, target_pos_r, callback=None):
    '''Wrapper around scipy fmin_bfgs'''
    # popt = fmin_bfgs(penalty, p0, callback=callback_4)
    return fmin_bfgs(
        penalty_func,
        q,
        args=(frame_l_id, frame_r_id, target_pos_l, target_pos_r),
        callback=callback
    )

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    
    # get end effector frame ids
    eff_l_name = LEFT_HAND
    frame_l_id = robot.model.getFrameId(eff_l_name)

    eff_r_name = RIGHT_HAND
    frame_r_id = robot.model.getFrameId(eff_r_name)

    cube_placement_l = getcubeplacement(cube, LEFT_HOOK)
    cube_placement_r = getcubeplacement(cube, RIGHT_HOOK)
    target_tf_l = cube_placement_l
    target_tf_r = cube_placement_r

    q = qcurrent.copy()
    max_samples = 2000
    tol = 5e-2
    tol_ori = 0.12  # orientation tolerance in radians (~7 deg)

    # quick check: if current config already valid and close enough, accept it
    try:
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        cur_pose_l = robot.data.oMf[frame_l_id]
        cur_pose_r = robot.data.oMf[frame_r_id]

        pos_ok = (norm(cur_pose_l.translation - target_tf_l.translation) < tol and
                  norm(cur_pose_r.translation - target_tf_r.translation) < tol)
        R_err_l = target_tf_l.rotation.dot(cur_pose_l.rotation.T)
        R_err_r = target_tf_r.rotation.dot(cur_pose_r.rotation.T)
        ori_ok = (norm(pin.log3(R_err_l)) < tol_ori and norm(pin.log3(R_err_r)) < tol_ori)

        if pos_ok and ori_ok and not collision(robot, q):
            return q, True
    except Exception:
        pass

    q = bfgs(penalty, q, frame_l_id, frame_r_id, target_tf_l, target_tf_r, callback=callback_viz if viz is not None else None)

    # check final result
    try:
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        final_pose_l = robot.data.oMf[frame_l_id]
        final_pose_r = robot.data.oMf[frame_r_id]

        pos_ok = (norm(final_pose_l.translation - target_tf_l.translation) < tol and
                  norm(final_pose_r.translation - target_tf_r.translation) < tol)
        R_err_l = target_tf_l.rotation.dot(final_pose_l.rotation.T)
        R_err_r = target_tf_r.rotation.dot(final_pose_r.rotation.T)
        ori_ok = (norm(pin.log3(R_err_l)) < tol_ori and norm(pin.log3(R_err_r)) < tol_ori)

        if pos_ok and ori_ok and not collision(robot, q):
            return q, True
    except Exception:
        pass

    # # random sampling around qcurrent (simple, robust approach)
    # rng = np.random.RandomState(0)
    # scales = np.linspace(0.35, 0.05, 8)
    # for scale in scales:
    #     for _ in range(int(max_samples / len(scales))):
    #         dq = rng.normal(scale=scale, size=q.shape)
    #         qc = q + dq
    #         qc = projecttojointlimits(robot, qc)

    #         try:
    #             pin.forwardKinematics(robot.model, robot.data, qc)
    #             pin.updateFramePlacements(robot.model, robot.data)
    #             pose_l = robot.data.oMf[frame_l_id]
    #             pose_r = robot.data.oMf[frame_r_id]
    #             updatevisuals(viz, robot, cube, qc)
    #         except Exception:
    #             continue

    #         if norm(pose_l.translation - target_pos_l) < tol and norm(pose_r.translation - target_pos_r) < tol:
    #             if not collision(robot, cube, qc):
    #                 return qc, True
    return robot.q0, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    # qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    # updatevisuals(viz, robot, cube, q0)
    print("Initial grasping pose success: ", successinit, " q0: ", q0)
    # print("End grasping pose success: ", successend)
    
    
    
