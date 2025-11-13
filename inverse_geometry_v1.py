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
from setup_meshcat import updatevisuals



# def cost(frame_l_id=None, frame_r_id=None, target_tf_l=None, target_tf_r=None):
#     # target_tf_* are pin.SE3 (or objects with .translation and .rotation)
#     pose_l = robot.data.oMf[frame_l_id]
#     pose_r = robot.data.oMf[frame_r_id]

#     # position error
#     e_pos_l = pose_l.translation - target_tf_l.translation
#     e_pos_r = pose_r.translation - target_tf_r.translation

#     # orientation error: relative rotation from current -> target, then log map to so(3)
#     R_err_l = target_tf_l.rotation.dot(pose_l.rotation.T)
#     R_err_r = target_tf_r.rotation.dot(pose_r.rotation.T)
#     e_ori_l = pin.log3(R_err_l)   # so(3) vector: axis * angle
#     e_ori_r = pin.log3(R_err_r)

#     # weights
#     w_pos = 10.0
#     w_ori = 1.0

#     error = 0.5 * ( w_pos*(norm(e_pos_l)**2 + norm(e_pos_r)**2) + w_ori*(norm(e_ori_l)**2 + norm(e_ori_r)**2) )
#     return error

def constraint(q):
    if collision(robot, q):
        res = 1.0
    else:
        res = 0
    return res

def get_min_collision_distance(robot, q):
    # 前向运动学更新
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data)
    
    # 计算所有几何体之间的最近距离
    pin.computeDistances(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
    
    # 获取最小距离
    min_distance = np.inf
    for d in robot.collision_data.distanceResults:
        if d.min_distance < min_distance:
            min_distance = d.min_distance
    return min_distance

# def penalty(q, frame_l_id=None, frame_r_id=None, target_tf_l=None, target_tf_r=None):
#     pin.forwardKinematics(robot.model, robot.data, q)
#     pin.updateFramePlacements(robot.model, robot.data)

#     return cost(frame_l_id, frame_r_id, target_tf_l, target_tf_r)# + 0.01*collision_severity# + 10 * constraint(q)

def penalty(q, frame_l_id, frame_r_id, target_tf_l, target_tf_r, q_ref=None):
    q = np.asarray(q).reshape(-1)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    # compute pos / ori errors as before
    pos_err = (norm(robot.data.oMf[frame_l_id].translation - target_tf_l.translation) +
               norm(robot.data.oMf[frame_r_id].translation - target_tf_r.translation)) * 0.5
    ori_err = (norm(pin.log3(target_tf_l.rotation.dot(robot.data.oMf[frame_l_id].rotation.T))) +
               norm(pin.log3(target_tf_r.rotation.dot(robot.data.oMf[frame_r_id].rotation.T)))) * 0.5

    # normalize by expected tolerances
    pos_scale = 0.05   # 5cm scale
    ori_scale = 0.1    # ~5.7 deg scale

    cost = (pos_err / pos_scale)**2 + 0.5 * (ori_err / ori_scale)**2

    # postural bias
    if q_ref is not None:
        cost += 0.01 * np.linalg.norm(q - q_ref)**2

    return cost

def callback_viz(q):
    updatevisuals(viz, robot, cube, q)

# def bfgs(penalty_func, q, frame_l_id, frame_r_id, target_pos_l, target_pos_r, callback=None):
#     '''Wrapper around scipy fmin_bfgs'''
#     # popt = fmin_bfgs(penalty, p0, callback=callback_4)
#     return fmin_bfgs(
#         penalty_func,
#         q,
#         args=(frame_l_id, frame_r_id, target_pos_l, target_pos_r),
#         callback=callback,
#         maxiter=5000
#     )


def nocollision_constraint(q):
    # 1 表示安全，-1 表示碰撞
    # if not collision(robot, q):
    #     return 1.0
    # else:
    min_distance = get_min_collision_distance(robot, q)
    return min_distance
    # if min_distance < 0:
    #     collision_severity = -min_distance  # positive value indicating penetration depth
    # else:
    #     collision_severity = 0.0
    # if collision(robot, q):
    #     return -1.0   # 不满足约束
    # else:
    #     return 1.0    # 满足约束（>0

from scipy.optimize import fmin_slsqp

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    pin.forwardKinematics(robot.model, robot.data, qcurrent)

    if viz is not None:
        updatevisuals(viz, robot, cube, qcurrent)
    
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
    tol = 1e-2
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

    min_penalty_threshold = 1e-3
    best_q = None
    best_cost = np.inf

    for i in range(5):  # try multiple times
        q_try = pin.randomConfiguration(robot.model)
        q_opt = fmin_slsqp(
            func=penalty,
            x0=q_try,
            f_ieqcons=lambda q, *args: np.array([nocollision_constraint(q)]),
            args=(frame_l_id, frame_r_id, target_tf_l, target_tf_r),
            acc=1e-3,
            iter=500,
            iprint=2
        )

        final_cost = penalty(q_opt, frame_l_id, frame_r_id, target_tf_l, target_tf_r)
        
        if final_cost < best_cost:
            best_q = q_opt
            best_cost = final_cost
        
        if best_cost < min_penalty_threshold:
            break
    
    q = best_q

    # min_penalty_threshold = 1e-3
    # best_q = None
    # best_cost = np.inf

    # for i in range(5):  # try multiple times
    #     q_try = pin.randomConfiguration(robot.model)
    #     q_opt = bfgs(penalty, q_try, frame_l_id, frame_r_id, target_tf_l, target_tf_r)
    #     final_cost = penalty(q_opt, frame_l_id, frame_r_id, target_tf_l, target_tf_r)
        
    #     if final_cost < best_cost:
    #         best_q = q_opt
    #         best_cost = final_cost
        
    #     if best_cost < min_penalty_threshold:
    #         break
    # # q = bfgs(penalty, q, frame_l_id, frame_r_id, target_tf_l, target_tf_r, callback=callback_viz if viz is not None else None)
    # q = best_q
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

        print("Final position error left:", norm(final_pose_l.translation - target_tf_l.translation))
        print("Final position error right:", norm(final_pose_r.translation - target_tf_r.translation))
        print("Final orientation error left (rad):", norm(pin.log3(R_err_l)))
        print("Final orientation error right (rad):", norm(pin.log3(R_err_r)))
        is_in_collision = collision(robot, q)
        print("Collision:", is_in_collision)
        if is_in_collision:
         for k in range(len(robot.collision_model.collisionPairs)): 
             cr = robot.collision_data.collisionResults[k]
             cp = robot.collision_model.collisionPairs[k]
             if cr.isCollision():
                 print("collision pair:",robot.collision_model.geometryObjects[cp.first].name,",",robot.collision_model.geometryObjects[cp.second].name,"- collision:","Yes" if cr.isCollision() else "No")
     
        print("pos_ok:", pos_ok, " ori_ok:", ori_ok)
        if pos_ok and ori_ok and not is_in_collision:
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
    return q, False#robot.q0

from pinocchio.utils import rotate
def sample_cube_placement():
    while True:
        # randomly sample a configuration
        # CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.3, 0.93]))
        # CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.4, 0.11, 0.93]))
        low_bound  = np.array([ 0,   -0.5,  0.93])
        high_bound = np.array([ 0.6,  0.5,   1.5])
        # rand_pos_cube = np.array([0.52998259, -0.0477236,  1.16745428])

        rand_pos_cube = np.random.uniform(low_bound, high_bound)

        new_cube_placement = pin.SE3(rotate('z', 0.), rand_pos_cube)
        setcubeplacement(robot, cube, new_cube_placement)

        # check if cube placement is valid (not in collision with obstacle or table)
        in_collision_cube = pin.computeCollisions(cube.model, cube.data, cube.collision_model, cube.collision_data, cube.q0, True) or pin.computeCollisions(robot.model, robot.data, robot.collision_model, robot.collision_data, robot.q0, True) #pin.computeCollisions(robot.model, robot.data, robot.collision_model, robot.collision_data, robot.q0, True)
        if in_collision_cube:
            print("Sampled cube placement in collision, resampling...")
            continue
        else:
            break
    return new_cube_placement

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()
    
    # q = robot.q0.copy()
    
    # # new_cube_placement = sample_cube_placement()
    # # print("Sampled cube placement: ", new_cube_placement.translation)
    # new_cube_placement = pin.SE3(rotate('z', 0),np.array([0.52998259, -0.0477236,  1.16745428]))
    # # q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    # qe,successend = computeqgrasppose(robot, q, cube, new_cube_placement,  viz)
    # updatevisuals(viz, robot, cube, qe)
    # # updatevisuals(viz, robot, cube, q0)
    # # print("Initial grasping pose success: ", successinit, " q0: ", q0)
    # print("End grasping pose success: ", successend)

    for i, name in enumerate(robot.model.names):
        print(f"Joint index: {i:2d}, name: {name}")

    
    
    
