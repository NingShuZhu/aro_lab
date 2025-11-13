#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv
from pinocchio.utils import rotate

from config import LEFT_HAND, RIGHT_HAND
import time
from solution import LMRREF, RMLREF
from tools import collision
from tools import setcubeplacement

def check_left_right_hand_relative_pose(robot, q):
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    pose_l = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    pose_r = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
    LWR = pose_l.inverse() * pose_r
    # RML = LWR.inverse()
    pos_err = np.linalg.norm(LWR.translation - LMRREF.translation)
    rot_err = np.linalg.norm(LWR.rotation - LMRREF.rotation)
    thresh_pos = 1e-3
    thresh_rot = 1e-3
    return (pos_err < thresh_pos) and (rot_err < thresh_rot)

# def sample_cube_placement():
#     while True:
#         # randomly sample a configuration
#         # CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.3, 0.93]))
#         # CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.4, 0.11, 0.93]))
#         low_bound  = np.array([ 0,   -0.5,  0.93])
#         high_bound = np.array([ 0.6,  0.5,   1.5])

#         rand_pos_cube = np.random.uniform(low_bound, high_bound)

#         new_cube_placement = pin.SE3(rotate('z', 0.), rand_pos_cube)
#         setcubeplacement(robot, cube, new_cube_placement)

#         # check if cube placement is valid (not in collision with obstacle or table)
#         in_collision_cube = pin.computeCollisions(cube.model, cube.data, cube.collision_model, cube.collision_data, cube.q0, True) and pin.computeCollisions(robot.model, robot.data, robot.collision_model, robot.collision_data, robot.q0, True)
#         if in_collision_cube:
#             print("Sampled cube placement in collision, resampling...")
#             continue
#         else:
#             break
#     return new_cube_placement

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
        in_collision_cube = pin.computeCollisions(cube.model, cube.data, cube.collision_model, cube.collision_data, cube.q0, True)
        if in_collision_cube:
            print("Sampled cube placement in collision, resampling...")
            continue
        else:
            break
    return new_cube_placement

def RAND_CONF_IK(qcurrent):
    # ramdomly samle the placement of the cube, then use IK to find a configuration, check constraints
    while True:
        new_cube_placement = sample_cube_placement()
        q, success = computeqgrasppose(robot, qcurrent, cube, new_cube_placement, viz=viz, turb_scale=0.35)
        if success:
            print("Found IK solution")
            return q
        else:
            print("no solution found")
        # setcubeplacement(robot, cube, new_cube_placement)
        # pin.updateFramePlacements(robot.model, robot.data)

        # use IK to find a configuration that grasps the cube
        # for _ in range(10):
        #     q0 = robot.q0.copy()
        #     q, success = computeqgrasppose(robot, q0, cube, new_cube_placement, viz=viz)
        #     if success:
        #         # check if the configuration is collision-free
        #         if not collision(robot, q):
        #             print("Found IK solution")
        #             return q
        #         else:
        #             print("IK solution in collision")
        

# def RAND_CONF(checkcollision=True):
#     '''
#     Return a random configuration, not in collision, satisfying relative grasping constraints
#     '''
#     while True:
#         q = pin.randomConfiguration(robot.model)  # sample between -3.2 and +3.2.
#         grasping = check_left_right_hand_relative_pose(robot, q)
#         if grasping and (not (checkcollision and collision(robot, q))): return q
        
# def distance(q1,q2):    
#     '''Return the euclidian distance between two configurations'''
#     return np.linalg.norm(q2-q1)
        
# def NEAREST_VERTEX(G,q_rand):
#     '''returns the index of the Node of G with the configuration closest to q_rand  '''
#     min_dist = 10e4
#     idx=-1
#     for (i,node) in enumerate(G):
#         dist = distance(node[1],q_rand) 
#         if dist < min_dist:
#             min_dist = dist
#             idx = i
#     return idx

def cube_distance(p1, p2, w_pos=1.0, w_rot=2.0):
    # 平移距离 (欧氏距离)
    pos_dist = np.linalg.norm(p1.translation - p2.translation)

    # 旋转距离：用 log3(R1.T * R2) 得到旋转误差向量（弧度）
    R_err = p1.rotation.T @ p2.rotation
    rot_vec = pin.log3(R_err)
    rot_dist = np.linalg.norm(rot_vec)

    # 综合距离（单位混合时建议加权）
    dist = w_pos * pos_dist + w_rot * rot_dist

    return dist

def nearest_cube_idx(G, pos_rand):
    min_dist = 10e4
    idx=-1
    for (i,node) in enumerate(G):
        dist = cube_distance(node[1],pos_rand) 
        if dist < min_dist:
            min_dist = dist
            idx = i
    return idx

# def ADD_EDGE_AND_VERTEX(G,parent,q):
#     G += [(parent,q)]

def ADD_EDGE_AND_VERTEX_cube(G, parent, pos, q):
    G += [(parent, pos, q)]

# def lerp(q0,q1,t):    
#     return q0 * (1 - t) + q1 * t

def lerp_pos(p0,p1,t): 
    trans = (1 - t) * p0.translation + t * p1.translation

    # --- 2. 旋转部分球面插值 (Slerp) ---
    q0 = pin.Quaternion(p0.rotation)
    q1 = pin.Quaternion(p1.rotation)
    q0.normalize()
    q1.normalize()
    q_interp = q0.slerp(t, q1)
    R_interp = q_interp.toRotationMatrix()

    # --- 3. 组合成新的SE3 ---
    return pin.SE3(R_interp, trans)   
    # return p0 * (1 - t) + p1 * t

# def NEW_CONF(q_near,q_rand,discretisationsteps, delta_q = None):
#     '''Return the closest configuration q_new such that the path q_near => q_new is the longest
#     along the linear interpolation (q_near,q_rand) that is collision free and of length <  delta_q'''
#     q_end = q_rand.copy()
#     dist = distance(q_near, q_rand)
#     if delta_q is not None and dist > delta_q:
#         #compute the configuration that corresponds to a path of length delta_q
#         q_end = lerp(q_near,q_rand,delta_q/dist)
#         # now dist == delta_q
#     dt = 1 / discretisationsteps
#     for i in range(1,discretisationsteps):
#         q = lerp(q_near,q_end,dt*i)
#         if collision(robot, q):
#             return lerp(q_near,q_end,dt*(i-1))
#     return q_end

def new_conf_cube(robot, q_ref, pos_near,pos_rand,discretisationdist, delta = None):
    '''Return the closest configuration q_new such that the path q_near => q_new is the longest
    along the linear interpolation (q_near,q_rand) that is collision free and of length <  delta'''
    pos_end = pos_rand.copy()
    dist = cube_distance(pos_near, pos_rand)
    print("dist: ", dist)
    if delta is not None and dist > delta:
        #compute the configuration that corresponds to a path of length delta
        pos_end = lerp_pos(pos_near,pos_rand,delta/dist)
        # now dist == delta
    # dt = 1 / discretisationsteps
    # for i in range(1,discretisationsteps):
    dt = discretisationdist
    n_steps = max(1, int(dist / dt))
    for i in range(1,n_steps):
        pos = lerp_pos(pos_near,pos_end,dt*i)
        setcubeplacement(robot, cube, pos)
        if pin.computeCollisions(cube.model, cube.data, cube.collision_model, cube.collision_data, cube.q0, True):
            return lerp_pos(pos_near,pos_end,dt*(i-1)), q_ref
        # check if the interpolated position is valid, i.e., corresponding q can be found
        q_new, success = computeqgrasppose(robot, q_ref, cube, pos, viz=viz, turb_scale=0.15)
        if not success or collision(robot, q_new):
            return lerp_pos(pos_near,pos_end,dt*(i-1)), q_ref
        else:
            q_ref = q_new.copy()
    return pos_end, q_ref

# def VALID_EDGE(q_new,q_goal,discretisationsteps):
#     return np.linalg.norm(q_goal -NEW_CONF(q_new, q_goal,discretisationsteps)) < 1e-3

    

# # RRT algorithm
# def rrt(q_init, q_goal, k, delta_q):
#     G = [(None,q_init)]
#     q0 = q_goal.copy()
#     for i in range(k):
#         print("k: ", i)
#         q_rand = RAND_CONF_IK(q0)   
#         q_near_index = NEAREST_VERTEX(G,q_rand)
#         q_near = G[q_near_index][1]        
#         q_new = NEW_CONF(q_near,q_rand,discretisationsteps_newconf, delta_q = delta_q)    
#         ADD_EDGE_AND_VERTEX(G,q_near_index,q_new)
#         if VALID_EDGE(q_new,q_goal,discretisationsteps_validedge):
#             print ("Path found!")
#             ADD_EDGE_AND_VERTEX(G,len(G)-1,q_goal)
#             return G, True
#     print("path not found")
#     return G, False

def rrt_cube(robot, q_init, q_end, pos_init, pos_end, k, discretisationdist, delta=None):
    G = [(None,pos_init, q_init)]
    i = 0
    while i < k:
        print("k: ", i)
        pos_rand = sample_cube_placement()
        
        q0 = G[-1][2]
        q, success = computeqgrasppose(robot, q0, cube, pos_rand, viz=viz, turb_scale=0.35)
        if not success:
            i = i-1
            continue
        else:
            # valid cube position
            # try to interpolate from existing nearest vertex
            pos_nearest_idx = nearest_cube_idx(G, pos_rand)
            pos_near = G[pos_nearest_idx][1]
            q_near = G[pos_nearest_idx][2]
            pos_new, q_new = new_conf_cube(robot, q_near, pos_near,pos_rand,discretisationdist, delta = delta)
            ADD_EDGE_AND_VERTEX_cube(G, pos_nearest_idx, pos_new, q_new)

            pos_try, _ = new_conf_cube(robot, q_new, pos_new, pos_end, discretisationdist)
            dist = cube_distance(pos_try, pos_end)
            if(dist < 1e-3): # i.e. pos_end is reachable
                print("path found!")
                ADD_EDGE_AND_VERTEX_cube(G, len(G)-1, pos_end, q_end)
                return G, True
        i = i+1

    print("path not found")
    return G, False


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    #TODO
    
    # G, foundpath = rrt(qinit, qgoal, k, delta_q)
    return [qinit, qgoal]
    pass


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)

# def visualize_rrt_in_meshcat(robot, cube, viz, G, sleep_time=0.05):
#     """
#     在 MeshCat 中依次显示 RRT 树节点（即 q 配置）对应的机器人姿态。
#     """
#     for i, (parent, q) in enumerate(G):
#         pin.forwardKinematics(robot.model, robot.data, q)
#         updatevisuals(viz, robot, cube, q)
#         print(f"Visualizing node {i}/{len(G)}")
#         time.sleep(sleep_time)

def visualize_rrt_in_meshcat(robot, cube, viz, G_cube, sleep_time=0.05):
    """
    在 MeshCat 中依次显示 RRT 树节点（即 q 配置）对应的机器人姿态。
    """
    for i, (parent, p, q) in enumerate(G_cube):
        print(f"cube {i}")
        setcubeplacement(robot, cube, p)
        # pin.forwardKinematics(robot.model, robot.data, q)
        updatevisuals(viz, robot, cube, q)
        print(f"Visualizing node {i}/{len(G_cube)}")
        time.sleep(sleep_time)

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from setup_meshcat import updatecubeframes, updatevisuals
    
    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6000")
    q0 = robot.q0.copy()

    # discretisationsteps = 10
    discretisationdist = 0.02
    k = 10

    qi,successinit = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT) #, viz
    qe,successend = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET) #, viz
    # updatevisuals(viz, robot, cube, qe)
    
    if not (successend and successinit):
        print ("error: invalid end configuration")
        exit(0)

    G_cube, foundpath = rrt_cube(robot, qi, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k, discretisationdist)

    print("found path? ", foundpath)

    visualize_rrt_in_meshcat(robot, cube, viz, G_cube, 1)
    # cube_pos = G_cube[-1][1]
    # q_last = G_cube[-1][2]
    # setcubeplacement(robot, cube, cube_pos)
    # updatevisuals(viz, robot, cube, q_last)
    # random configuration sampling test
    # for _ in range(10):
    #     q = RAND_CONF_IK(q0)
    #     updatevisuals(viz, robot, cube, q)
    #     time.sleep(1)
    #     q0 = q.copy()

    # # parameters for the RRT
    # discretisationsteps_newconf = 200 #To tweak later on
    # discretisationsteps_validedge = 200 #To tweak later on
    # k = 10  #To tweak later on
    # delta_q = 3. #To tweak later on
    
    # # q = robot.q0.copy()
    # # get initial and goal configurations under grasping constraints
    # qi,successinit = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT) #, viz
    # qe,successend = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET, viz) #, viz
    # updatevisuals(viz, robot, cube, qe)
    # # if not successinit:
    # #     print ("error: invalid initial configuration")
    # #     exit(0)
    # if not (successend and successinit):
    #     print ("error: invalid end configuration")
    #     exit(0)

    # G, foundpath = rrt(qi, qe, k, delta_q)
    # print("foundPath: ", foundpath)
    # print("G: ", G)
    # time.sleep(2)

    # print("now see the graph qs'")
    # visualize_rrt_in_meshcat(robot, cube, viz, G, 1)
    # path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    # displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
