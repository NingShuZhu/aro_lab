import numpy as np
import pinocchio as pin

from tools import setupwithmeshcat
from setup_meshcat import updatevisuals
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from tools import collision, jointlimitsviolated, getcubeplacement, setcubeplacement, projecttojointlimits


def inverse_kinematics(model, data, frame_name, target_SE3, q_init, max_iter=100, eps=1e-4, alpha=0.5):
    """
    使用雅可比迭代法求解逆运动学
    :param model: Pinocchio 模型
    :param data: Pinocchio 数据结构
    :param frame_name: 末端 frame 名称，例如 "RARM_JOINT7_Link"
    :param target_SE3: 目标末端位姿 (pin.SE3)
    :param q_init: 初始关节角
    :param max_iter: 最大迭代次数
    :param eps: 收敛阈值
    :param alpha: 步长
    """
    q = q_init.copy()
    frame_id = model.getFrameId(frame_name)

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        current_SE3 = data.oMf[frame_id]
        err_vec = pin.log6(current_SE3.inverse() * target_SE3)

        if np.linalg.norm(err_vec) < eps:
            print(f"Converged in {i} iterations")
            return q, True

        # 计算雅可比
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
        v = alpha * np.linalg.pinv(J) @ err_vec  # damped least squares 可改进
        q = pin.integrate(model, q, v)

    print("IK did not converge")
    return q, False

def inverse_kinematics_dual(robot,
                             frame_names, target_SE3s,
                             q_init, max_iter=200, eps=1e-4, alpha=0.5):
    """
    同时满足多个末端（例如左右手）的逆运动学
    :param frame_names: list[str]，末端名称列表，如 ["LARM_JOINT7_Link", "RARM_JOINT7_Link"]
    :param target_SE3s: list[pin.SE3]，目标位姿列表
    """
    q = q_init.copy()
    frame_ids = [robot.model.getFrameId(name) for name in frame_names]

    for i in range(max_iter):
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)

        # ---- 拼接误差向量 ----
        errors = []
        for fid, target in zip(frame_ids, target_SE3s):
            current = robot.data.oMf[fid]
            err_vec = pin.log6(current.inverse() * target)
            errors.append(err_vec)
        err_vec = np.concatenate(errors)

        if np.linalg.norm(err_vec) < eps:
            # print(f"✅ Dual-IK converged in {i} iterations")
            return q, True

        # ---- 拼接雅可比矩阵 ----
        J_blocks = []
        for fid in frame_ids:
            J = pin.computeFrameJacobian(robot.model, robot.data, q, fid, pin.ReferenceFrame.LOCAL)
            J_blocks.append(J)
        J_full = np.vstack(J_blocks)

        # ---- 最小二乘解 Δq ----
        dq = alpha * np.linalg.pinv(J_full) @ err_vec
        q = pin.integrate(robot.model, q, dq)

        q = projecttojointlimits(robot,q)

    # print("❌ Dual-IK did not converge")
    return q, False


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    cube_placement_l = getcubeplacement(cube, LEFT_HOOK)
    cube_placement_r = getcubeplacement(cube, RIGHT_HOOK)

    q_init = qcurrent.copy()

    q_sol, success = inverse_kinematics_dual(robot, [LEFT_HAND, RIGHT_HAND], [cube_placement_l, cube_placement_r], q_init)
    
    if success:
        print("IK solution found:", q_sol)
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

    

robot, cube, viz = setupwithmeshcat()

q = robot.q0.copy()

# 末端目标位姿
# setcubeplacement(robot, cube, CUBE_PLACEMENT_TARGET)
# cube_placement_l = getcubeplacement(cube, LEFT_HOOK)
# cube_placement_r = getcubeplacement(cube, RIGHT_HOOK)
# 初始姿态
q_init = np.zeros(robot.nq)

# # 求解
# q_sol, success = inverse_kinematics(robot.model, robot.data, LEFT_HAND, cube_placement_l, q_init)

# if success:
#     print("IK solution found:", q_sol)
# else:
#     print("Failed to converge")

# updatevisuals(viz, robot, cube, q_sol)

# q_sol, success = inverse_kinematics_dual(robot.model, robot.data, [LEFT_HAND, RIGHT_HAND], [cube_placement_l, cube_placement_r], q_init)

# if success:
#     print("IK solution found:", q_sol)
# else:
#     print("Failed to converge")

# updatevisuals(viz, robot, cube, q_sol)

q_sol, success = computeqgrasppose(robot, q_init, cube, CUBE_PLACEMENT_TARGET, viz)