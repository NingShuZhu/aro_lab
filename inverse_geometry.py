import numpy as np
import pinocchio as pin

from tools import setupwithmeshcat
from setup_meshcat import updatevisuals
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from tools import collision, jointlimitsviolated, getcubeplacement, setcubeplacement, projecttojointlimits
from tools import setcubeplacement

def inverse_kinematics_dual(robot, frame_names, target_SE3s, q_init, max_iter=200, eps=1e-4, alpha=0.5):
    """
    solve inverse kinematics satisfying the postions of both the two end-effectors
    :param frame_names: a list of names of the two end-effectors ([LEFT_HAND, RIGHT_HAND])
    :param target_SE3s: a list of target positions (pin.SE3) for the two ee's
    :param q_init: the initial q to start searching
    """
    q = q_init.copy()
    frame_ids = [robot.model.getFrameId(name) for name in frame_names]

    for i in range(max_iter):
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)

        # compute and concatenate the error vectors (between current poses and the targets)
        errors = []
        for fid, target in zip(frame_ids, target_SE3s):
            current = robot.data.oMf[fid]
            err_vec = pin.log6(current.inverse() * target) # 6
            errors.append(err_vec)
        err_vec = np.concatenate(errors) # len: 12

        if np.linalg.norm(err_vec) < eps:
            # print(f"Dual-IK converged in {i} iterations")
            return q, True

        # compute and combine the Jacobian matrices
        J_blocks = []
        for fid in frame_ids:
            J = pin.computeFrameJacobian(robot.model, robot.data, q, fid, pin.ReferenceFrame.LOCAL) # 6, n
            J_blocks.append(J)
        J_full = np.vstack(J_blocks) # shape: 12, n

        # least square, dq = step * (Moore-Penrose pseudo-inverse(J)) @ error
        dq = alpha * np.linalg.pinv(J_full) @ err_vec
        q = pin.integrate(robot.model, q, dq)

        q = projecttojointlimits(robot,q)

    # print("Dual-IK did not converge")
    return q, False


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    cube_placement_l = getcubeplacement(cube, LEFT_HOOK)
    cube_placement_r = getcubeplacement(cube, RIGHT_HOOK)

    q_init = qcurrent.copy()

    q_sol, success = inverse_kinematics_dual(robot, [LEFT_HAND, RIGHT_HAND], [cube_placement_l, cube_placement_r], q_init)
    
    if success:
        print("IK solution found") #, q_sol
    else:
        print("Failed to converge")
        return q_sol, False

    # collision check
    in_collision = collision(robot, q_sol)
    if in_collision:
        print("collision!")

    # joint limitation check
    joint_not_ok = jointlimitsviolated(robot,q_sol)
    if joint_not_ok:
        print("joint limits voilated!")

    if viz is not None:
        updatevisuals(viz, robot, cube, q_sol)

    if in_collision or joint_not_ok:
        return q_sol, False
    
    return q_sol, True

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    # new_cube_placement = sample_cube_placement()
    # print("Sampled cube placement: ", new_cube_placement.translation)
    # qe,successend = computeqgrasppose(robot, q, cube, new_cube_placement,  viz)

    # q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, qe)
    # updatevisuals(viz, robot, cube, q0)
    # print("Initial grasping pose success: ", successinit, " q0: ", q0)
    print("End grasping pose success: ", successend)
    
    