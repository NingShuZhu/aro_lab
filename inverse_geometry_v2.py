#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitscost #distanceToObstacle
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement
from scipy.optimize import fmin_bfgs,fmin_slsqp
from setup_meshcat import updatevisuals


def constraint(q):
    if collision(robot, q):
        res = -1.0
    else:
        res = 1.0
    return res

# def distanceToObstacle_my(q):
#     """Return signed distance: positive = safe, negative = penetration."""
#     pin.forwardKinematics(robot.model, robot.data, q)
#     pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
#     pin.computeCollisions(robot.model, robot.data, robot.collision_model, robot.collision_data, q, True)

#     geomidobs = robot.collision_model.getGeometryId('obstaclebase_0')
#     geomidtable = robot.collision_model.getGeometryId('baseLink_0')
#     pairs = [i for i, pair in enumerate(robot.collision_model.collisionPairs)
#              if pair.second == geomidobs or pair.second == geomidtable]

#     min_dist = np.inf
#     for idx in pairs:
#         dist_result = pin.computeDistance(robot.collision_model, robot.collision_data, idx)
#         d = dist_result.min_distance
#         # 如果已经检测到碰撞，则返回负值
#         if robot.collision_data.collisionResults[idx].isCollision():
#             d = -abs(d)
#         if d < min_dist:
#             min_dist = d
#     return min_dist

# def distanceToObstacle_my(q, collision_names=None):
#     """
#     Return signed distance between robot and obstacle(s):
#       >0 : minimal separation distance (safe)
#       0  : touching
#       <0 : collision (we return negative value as an indicator; may be approximate)
#     collision_names: list of geometry names considered as obstacles (optional).
#     """
#     q = np.asarray(q, dtype=float).reshape(-1)

#     # 1) update kinematics & geometry with q (use overload with q)
#     pin.forwardKinematics(robot.model, robot.data, q)
#     pin.updateFramePlacements(robot.model, robot.data)
#     pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)

#     # 2) compute distances and collisions (ensure both are executed)
#     # computeDistances fills collision_data.distanceResults
#     pin.computeDistances(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
#     # computeCollisions fills collision_data.collisionResults and returns True if any collision
#     in_collision = pin.computeCollisions(robot.model, robot.data, robot.collision_model, robot.collision_data, q, True)

#     # 3) filter the pairs of interest (optional)
#     if collision_names is None:
#         # default: consider all pairs
#         distance_results = robot.collision_data.distanceResults
#         collision_results = robot.collision_data.collisionResults
#     else:
#         # build indices for geometry pairs that involve any name in collision_names
#         name_set = set(collision_names)
#         selected_pairs = []
#         for i, pair in enumerate(robot.collision_model.collisionPairs):
#             name_first = robot.collision_model.geometryObjects[pair.first].name
#             name_second = robot.collision_model.geometryObjects[pair.second].name
#             if (name_first in name_set) or (name_second in name_set):
#                 selected_pairs.append(i)
#         distance_results = [robot.collision_data.distanceResults[i] for i in selected_pairs]
#         collision_results = [robot.collision_data.collisionResults[i] for i in selected_pairs]

#     # 4) get minimal distance among considered pairs (distanceResults.min_distance are >= 0)
#     min_d = np.inf
#     for dr in distance_results:
#         d = dr.min_distance
#         if d < min_d:
#             min_d = d

#     # 5) if any of the considered collisionResults indicates collision -> return negative value
#     collided = any(cr.isCollision() for cr in collision_results)
#     if collided:
#         # Option A: return negative of the minimal distance (if min_d computed, may be 0)
#         # Option B: return -eps to indicate penetration (choose whichever is meaningful)
#         # Here we return -max(min_d, 1e-3) so we get a negative number and somewhat informative magnitude.
#         return -max(min_d, 1e-3)
#     else:
#         # safe: return minimal positive distance (could be inf if no distanceResults)
#         return min_d if np.isfinite(min_d) else np.inf



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
    pos_scale = 0.01   # 5cm scale
    ori_scale = 0.05    # ~5.7 deg scale

    cost = (pos_err / pos_scale)**2 + 0.5 * (ori_err / ori_scale)**2

    # joint limits
    cost += 100 * jointlimitscost(robot, q)

    # # postural bias
    # # if q_ref is not None:
    # #     cost += 0.01 * np.linalg.norm(q - q_ref)**2
    # if q_ref is not None:
    #     # weights = np.ones_like(q)
    #     # weights[0] = 6.0   # 第一个关节权重最大
    #     weight = 10
    #     diff = q - q_ref
    #     cost += 0.01 * diff[0] * weight#np.sum(weights * diff**2)

    # collision penalty:
    cd = constraint(q) #distanceToObstacle_my(q)
    if cd < 0:
        cost -= 10000*cd

    return cost

def penalty_reduced(q_opt_vars, frame_l_id, frame_r_id, target_tf_l, target_tf_r, q_ref, opt_idx):
    # reconstruct q from the frozen variants
    q_full = q_ref.copy()
    q_full[opt_idx] = q_opt_vars
    return penalty(q_full, frame_l_id, frame_r_id, target_tf_l, target_tf_r, q_ref)

def callback_viz(q):
    updatevisuals(viz, robot, cube, q)


# def nocollision_constraint(q):
#     margin=0.01
#     min_distance = distanceToObstacle(robot, q)
#     return min_distance - margin

def nc_reduced(q_opt_vars, frame_l_id, frame_r_id, target_tf_l, target_tf_r, q_ref, opt_idx):
    q_full = q_ref.copy()
    q_full[opt_idx] = q_opt_vars
    return np.array([constraint(q_full)])#nocollision_constraint(q) distanceToObstacle_my(q_full)


from scipy.optimize import fmin_slsqp

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None, turb_scale=0.35):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    pin.forwardKinematics(robot.model, robot.data, qcurrent)

    q_ref = qcurrent.copy()

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

    # freeze the head joints
    frozen_idx = [1, 2]
    opt_idx = [i for i in range(len(q)) if i not in frozen_idx]
    
    
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

    min_penalty_threshold = 0.1
    best_q = None
    best_cost = np.inf
    rng = np.random.RandomState(0)

    for i in range(10):  # try multiple times
        # random sampling around qcurrent (simple, robust approach)
        while True:
            if i == 0:
                q_try = q.copy()
                break
            dq = rng.normal(scale=turb_scale, size=q.shape)
            # freeze the chest joint and the head joints
            dq[1:3] = 0.0
            qc = q + dq
            q_try = projecttojointlimits(robot, qc)
            pin.forwardKinematics(robot.model, robot.data, q_try)
            if not collision(robot, q_try):
                break
            else:
                print("collision, resample q_try")

        x_try = q_try[opt_idx]
        print(len(x_try))

        # q_try = pin.randomConfiguration(robot.model)
        # visualize random q
        pin.forwardKinematics(robot.model, robot.data, q_try)
        updatevisuals(viz, robot, cube, q_try)

        q_opt_vars = fmin_slsqp(
            func=penalty_reduced,
            x0=x_try,
            #f_ieqcons=nc_reduced,#lambda q, *args: np.array([nocollision_constraint(q, q_ref)]),
            args=(frame_l_id, frame_r_id, target_tf_l, target_tf_r, q_ref, opt_idx),
            acc=1e-3,
            iter=100,
            iprint=2
        )

        q_opt = q.copy()
        q_opt[opt_idx] = q_opt_vars

        final_cost = penalty(q_opt, frame_l_id, frame_r_id, target_tf_l, target_tf_r)
        print("penalty value: ", final_cost)

        if final_cost < best_cost:
            best_q = q_opt
            best_cost = final_cost
        
        if best_cost < min_penalty_threshold:
            break
    
    q = best_q

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
     
        # print("min collision distance: ", distanceToObstacle_my(q))
        print("pos_ok:", pos_ok, " ori_ok:", ori_ok)
        if pos_ok and ori_ok and not is_in_collision:
            return q, True
    except Exception:
        pass

    return q, False#robot.q0

from pinocchio.utils import rotate
def sample_cube_placement():
    while True:
        # randomly sample a configuration
        # CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.3, 0.93]))
        # CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.4, 0.11, 0.93]))
        low_bound  = np.array([ 0.2,   -0.3,  0.93])
        high_bound = np.array([ 0.6,  0.3,   1.2])
        # rand_pos_cube = np.array([0.52998259, -0.0477236,  1.16745428])

        rand_pos_cube = np.random.uniform(low_bound, high_bound)

        new_cube_placement = pin.SE3(rotate('z', 0.), rand_pos_cube)
        setcubeplacement(robot, cube, new_cube_placement)

        # check if cube placement is valid (not in collision with obstacle or table)
        in_collision_cube = pin.computeCollisions(cube.model, cube.data, cube.collision_model, cube.collision_data, cube.q0, True)# or pin.computeCollisions(robot.model, robot.data, robot.collision_model, robot.collision_data, robot.q0, True) #pin.computeCollisions(robot.model, robot.data, robot.collision_model, robot.collision_data, robot.q0, True)
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
    
    q = robot.q0.copy()
    
    new_cube_placement = sample_cube_placement()
    print("Sampled cube placement: ", new_cube_placement.translation)
    # new_cube_placement = pin.SE3(rotate('z', 0),np.array([0.52998259, -0.0477236,  1.16745428]))
    # q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    # qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    qe,successend = computeqgrasppose(robot, q, cube, new_cube_placement,  viz)

    
    updatevisuals(viz, robot, cube, qe)
    # updatevisuals(viz, robot, cube, q0)
    # print("Initial grasping pose success: ", successinit, " q0: ", q0)
    print("End grasping pose success: ", successend)
    
    # for i, name in enumerate(robot.model.names):
    #     print(f"Joint index: {i:2d}, name: {name}")
    
