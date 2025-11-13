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

from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from config import LEFT_HAND, RIGHT_HAND
import time
from solution import LMRREF, RMLREF
from tools import collision
from tools import setcubeplacement

def sample_between(start, goal, low_z, high_z, margin=0.05):
    p_start = start.translation
    p_goal = goal.translation
    x_min, x_max = sorted([p_start[0], p_goal[0]])
    y_min, y_max = sorted([p_start[1], p_goal[1]])
    z_min, z_max = sorted([p_start[2], p_goal[2]])
    
    # 给一点边界余量 margin
    x_min -= margin; x_max += margin
    y_min -= margin; y_max += margin
    z_min -= margin; z_max += (margin + high_z)
    z_min += low_z
    
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(y_min, y_max)
    z = np.random.uniform(z_min, z_max)
    return np.array([x, y, z])


def sample_cube_placement():
    while True:
        # randomly sample a configuration
        # CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.3, 0.93]))
        # CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.4, 0.11, 0.93]))
        # low_bound  = np.array([ 0.2,   -0.3,  0.93])
        # high_bound = np.array([ 0.6,  0.3,   1.2])
        # rand_pos_cube = np.array([0.52998259, -0.0477236,  1.16745428])

        rand_pos_cube = sample_between(CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, 0.2, 0.5)#np.random.uniform(low_bound, high_bound)

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


def ADD_EDGE_AND_VERTEX_cube(G, parent, pos, q):
    G += [(parent, pos, q)]


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


def new_conf_cube(robot, q_ref, pos_near,pos_rand,discretisationdist, delta = None):
    '''Return the closest configuration q_new such that the path q_near => q_new is the longest
    along the linear interpolation (q_near,q_rand) that is collision free and of length <  delta'''
    # q0 = robot.q0.copy()
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
            if i > 1:
                return lerp_pos(pos_near,pos_end,dt*(i-1)), q_ref
            else:
                return None, None
        # check if the interpolated position is valid, i.e., corresponding q can be found
        q_new, success = computeqgrasppose(robot, q_ref, cube, pos, viz=viz)
        if not success or collision(robot, q_new):
            if i > 1:
                return lerp_pos(pos_near,pos_end,dt*(i-1)), q_ref
            else:
                return None, None
        else:
            q_ref = q_new.copy()
    return pos_end, q_ref


def rrt_cube(robot, cube, q_init, q_end, pos_init, pos_end, k, discretisationdist, delta=.3):
    G = [(None,pos_init, q_init)]
    # G_inter = [(None,pos_init, q_init)] # store the interpolated positions and q's
    i = 0
    while i < k:
        print("k: ", i)
        pos_rand = sample_cube_placement()
        
        q0 = G[-1][2]#robot.q0.copy()
        q, success = computeqgrasppose(robot, q0, cube, pos_rand, viz=viz)
        if not success:
            continue
        else:
            # valid cube position
            # try to interpolate from existing nearest vertex
            pos_nearest_idx = nearest_cube_idx(G, pos_rand)
            pos_near = G[pos_nearest_idx][1]
            q_near = G[pos_nearest_idx][2]
            pos_new, q_new = new_conf_cube(robot, q_near, pos_near,pos_rand,discretisationdist, delta = delta)
            if pos_new == None:
                continue
            ADD_EDGE_AND_VERTEX_cube(G, pos_nearest_idx, pos_new, q_new)
        
            pos_try, _ = new_conf_cube(robot, q_new, pos_new, pos_end, discretisationdist)
            
            if pos_try != None:
                dist = cube_distance(pos_try, pos_end)
                if(dist < 1e-3): # i.e. pos_end is reachable
                    print("path found!")
                    ADD_EDGE_AND_VERTEX_cube(G, len(G)-1, pos_end, q_end)
                    # add_vertices_to_graph(vertices2end, len(G_inter)-1, G_inter)
                    return G, True
        i = i+1

    print("path not found")
    return G, False


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(robot, cube, qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    #TODO
    G_cube, foundpath = rrt_cube(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, k, 0.01)
    # G, foundpath = rrt(qinit, qgoal, k, delta_q)
    return [qinit, qgoal]
    pass


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


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
    from setup_meshcat import updatevisuals
    
    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6000")
    q0 = robot.q0.copy()

    # discretisationsteps = 10
    discretisationdist = 0.001
    k = 10

    qi,successinit = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT) #, viz
    qe,successend = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET) #, viz
    # updatevisuals(viz, robot, cube, qe)
    
    if not (successend and successinit):
        print ("error: invalid end configuration")
        exit(0)

    G_cube, foundpath = rrt_cube(robot, cube, qi, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, k, discretisationdist)

    print("found path? ", foundpath)

    print("G_cube: ")
    visualize_rrt_in_meshcat(robot, cube, viz, G_cube, 1)
