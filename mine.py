from tools import setupwithmeshcat
robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6000")

from config import LEFT_HAND, RIGHT_HAND

print ("Left hand joint name: ", LEFT_HAND)


import pinocchio as pin
q = robot.q0.copy()

#update the frame positions in robot.data given q
pin.framesForwardKinematics(robot.model,robot.data,q)

#now let's print the placement attached to the right hand
print ("Left hand joint placement: ")
pin.computeJointJacobians(robot.model,robot.data,q)
frameid = robot.model.getFrameId(LEFT_HAND)
oMframe = robot.data.oMf[frameid] 
print(oMframe)