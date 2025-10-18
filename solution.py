#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:43:26 2023

@author: stonneau
"""

from os.path import dirname, join, abspath
from time import sleep
import numpy as np
import pinocchio as pin #the pinocchio library
from pinocchio.utils import rotate
from tools import getcubeplacement
from tools import setupwithmeshcat
from config import LEFT_HOOK, RIGHT_HOOK, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
import inverse_geometry
from inverse_geometry import computeqgrasppose

robot, cube, viz = setupwithmeshcat()
inverse_geometry.robot = robot
inverse_geometry.cube = cube
inverse_geometry.viz = viz

#helpers 
#if needed, you can store the placement of the right hand in the left hand frame here
cube_placement_l = getcubeplacement(cube, LEFT_HOOK)
cube_placement_r = getcubeplacement(cube, RIGHT_HOOK)
target_tf_l = cube_placement_l
target_tf_r = cube_placement_r

LMRREF = cube_placement_l.inverse() * cube_placement_r
RMLREF = LMRREF.inverse()

print("LMRREF translation: ", LMRREF.translation)
print("LMRREF rotation matrix: \n", LMRREF.rotation)

q = robot.q0.copy()
    
q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
sleep(1)
qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)