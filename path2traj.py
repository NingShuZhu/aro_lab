import numpy as np
from bezier import Bezier

from scipy.special import comb

import numpy as np
from scipy.special import comb

def bezier_QP_trajectory_improved(path, total_time, degree=7, lam_smooth=1e-4,
                                  discretisation_t_by_arclength=True,
                                  weights=None,
                                  force_through_indices=None):
    """
    Improved Bezier QP fitting:
      - arc-length parameterization (optional)
      - weighted least-squares (weights per sample)
      - small smoothing regularizer (lam_smooth)
      - optional force_through_indices: list of sample indices to enforce exactly (hard)
    Inputs:
      path: (N, d) array-like, samples in the space you want to fit (ideally task-space positions)
      total_time: total duration (seconds)
      degree: Bezier degree (recommend between 5~12 depending on N)
      lam_smooth: smoothing weight (smaller -> closer to points)
      weights: None or array shape (N,) of positive weights (higher -> fit better)
      force_through_indices: None or list of indices (e.g. [0, N-1] to force endpoints)
    Returns:
      trajectory(t) -> (q, qdot, qddot)
    """
    path = np.asarray(path, dtype=float)
    N, d = path.shape
    if degree < 1:
        raise ValueError("degree must be >=1")
    if N < 2:
        raise ValueError("need at least two path points")

    # 1) parameterization t in [0,1] by arclength or uniform
    if discretisation_t_by_arclength:
        seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_len = seg_len.sum()
        if total_len <= 0:
            t_samples = np.linspace(0.0, 1.0, N)
        else:
            cum = np.concatenate(([0.0], np.cumsum(seg_len)))
            t_samples = cum / cum[-1]
    else:
        t_samples = np.linspace(0.0, 1.0, N)

    # 2) Build basis matrix M (N x (degree+1))
    M = np.zeros((N, degree + 1))
    for k, u in enumerate(t_samples):
        for i in range(degree + 1):
            M[k, i] = comb(degree, i) * (u ** i) * ((1 - u) ** (degree - i))

    # 3) smoothing regularizer D (second diff)
    if degree >= 2:
        D = np.zeros((degree - 1, degree + 1))
        for i in range(degree - 1):
            D[i, i:i+3] = [1.0, -2.0, 1.0]
    else:
        D = np.zeros((0, degree + 1))

    # 4) weights
    if weights is None:
        W = np.eye(N)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        assert w.shape[0] == N
        W = np.diag(w)

    # 5) Build normal equations (weighted)
    # Solve (M^T W M + lam * D^T D) P = M^T W X
    A = M.T @ W @ M
    if D.shape[0] > 0 and lam_smooth > 0:
        A += lam_smooth * (D.T @ D)
    B = M.T @ W @ path  # shape (degree+1, d)

    # 6) If no hard constraints, solve directly
    P = None
    if not force_through_indices:
        P = np.linalg.solve(A, B)
    else:
        # Hard-constrain certain sample indices: impose M_rows * P = X_rows for those indices
        # We'll eliminate constrained DOFs by rewriting the linear system.
        # Let rows C be indices constrained: M_C P = X_C
        C = np.array(sorted(force_through_indices), dtype=int)
        # Build constraint matrix for P: C_mat @ P = X_C, where C_mat = M[C,:] (shape len(C) x (deg+1))
        C_mat = M[C, :]            # (nc, deg+1)
        Xc = path[C, :]           # (nc, d)

        # We'll solve by partitioning unknown P into free and fixed via QR on C_mat^T
        # Simpler approach: find nullspace of C_mat and particular solution:
        # P = P_part + N * z, where C_mat @ P_part = Xc and C_mat @ N = 0
        # compute least-norm P_part via pseudoinverse:
        P_part_flat = np.linalg.pinv(C_mat) @ Xc  # (deg+1, d) minimal-norm solution of C_mat P = Xc

        # compute nullspace basis of C_mat (rows space) -> nullspace of shape ((deg+1) x k)
        # Use SVD to get nullspace
        U, S, Vt = np.linalg.svd(C_mat, full_matrices=True)
        rank = np.sum(S > 1e-12)
        nullspace = Vt[rank:].T  # shape ((deg+1) x (deg+1-rank))
        # Now param P = P_part_flat + nullspace @ z
        # Plug into normal equations: A (P_part + N z) = B  => (A N) z = B - A P_part
        if nullspace.size == 0:
            # no nullspace, P is fully constrained by C_mat; just use P_part_flat
            P = P_part_flat
        else:
            AN = A @ nullspace
            rhs = B - A @ P_part_flat
            # Solve for z (least-squares)
            z, *_ = np.linalg.lstsq(AN, rhs, rcond=None)
            P = P_part_flat + nullspace @ z

    # 7) Optionally enforce exact endpoints and zero velocities (repeat endpoints if desired)
    # If you want qdot(0)=0 and qdot(1)=0: enforce P1=P0 and P_{n-1}=P_n
    # We'll do it by overwriting control points if requested (simple):
    # (You can also put these as hard constraints in force_through_indices)
    P[0, :] = path[0, :]
    P[-1, :] = path[-1, :]
    # Enforce zero velocities at endpoints by duplication:
    P[1, :] = P[0, :].copy()
    P[-2, :] = P[-1, :].copy()

    # Construct Bezier objects (assume your Bezier class takes list of control points)
    bezier_curve = Bezier([P[i, :] for i in range(P.shape[0])], t_min=0.0, t_max=total_time)
    bezier_d1 = bezier_curve.derivative(1)
    bezier_d2 = bezier_curve.derivative(2)

    def trajectory(t):
        if t <= 0:
            return path[0, :].copy(), np.zeros(d), np.zeros(d)
        if t >= total_time:
            return path[-1, :].copy(), np.zeros(d), np.zeros(d)
        q = bezier_curve(t)
        qdot = bezier_d1(t)
        qddot = bezier_d2(t)
        return q, qdot, qddot

    return trajectory


def piecewise_bezier_trajectory(path, total_time,
                                degree=5,
                                lam_smooth=1e-4,
                                continuity='C1',   # 'C0', 'C1', or 'C2'
                                zero_end_vel=True):
    """
    Piecewise Bezier trajectory smoothing.
    Each segment is a Bezier curve, stitched with C1/C2 continuity.
    Args:
        path: (N, d) numpy array of key points.
        total_time: total duration (s)
        degree: degree of each Bezier segment (typically 3~7)
        lam_smooth: smoothing weight
        continuity: 'C0', 'C1', or 'C2'
        zero_end_vel: if True, enforce qdot(0)=qdot(T)=0
    Returns:
        trajectory(t) -> (q, qdot, qddot)
    """
    path = np.asarray(path, dtype=float)
    N, d = path.shape
    n_seg = N - 1
    seg_time = total_time / n_seg

    # Each segment has degree+1 control points
    P_all = np.zeros((n_seg, degree + 1, d))

    # --- Step 1: initial guess: straight line control points
    for i in range(n_seg):
        for k in range(degree + 1):
            alpha = k / degree
            P_all[i, k] = (1 - alpha) * path[i] + alpha * path[i + 1]

    # --- Step 2: enforce continuity constraints ---
    # We'll optimize intermediate control points with least squares + smoothing
    # For simplicity, we just solve for control points that minimize smoothness deviation

    # Smoothness regularization between adjacent segments
    for i in range(1, n_seg):
        # continuity constraints
        if continuity in ['C1', 'C2']:
            # derivative continuity at junction
            # qdot at end of segment i-1 = qdot at start of segment i
            P_prev = P_all[i - 1]
            P_curr = P_all[i]
            # derivative for Bezier: qdot(1) ∝ P_d - P_{d-1}, qdot(0) ∝ P_1 - P_0
            v_end_prev = P_prev[-1] - P_prev[-2]
            v_start_curr = P_curr[1] - P_curr[0]
            avg_v = 0.5 * (v_end_prev + v_start_curr)
            P_prev[-2] = P_prev[-1] - avg_v
            P_curr[1] = P_curr[0] + avg_v

        if continuity == 'C2' and degree >= 3:
            # acceleration continuity
            a_end_prev = P_prev[-1] - 2 * P_prev[-2] + P_prev[-3]
            a_start_curr = P_curr[2] - 2 * P_curr[1] + P_curr[0]
            avg_a = 0.5 * (a_end_prev + a_start_curr)
            P_prev[-3] = P_prev[-2] - avg_a / 2
            P_curr[2] = P_curr[1] + avg_a / 2

    # --- Step 3: optional zero end velocities
    if zero_end_vel:
        P_all[0, 1] = P_all[0, 0]  # qdot(0)=0
        P_all[-1, -2] = P_all[-1, -1]  # qdot(T)=0

    # # --- Step 4: smooth within each segment (optional lam_smooth)
    # if lam_smooth > 0:
    #     for i in range(n_seg):
    #         D = np.zeros((degree - 1, degree + 1))
    #         for j in range(degree - 1):
    #             D[j, j:j+3] = [1, -2, 1]
    #         Q = np.eye(degree + 1) + lam_smooth * (D.T @ D)
    #         B = P_all[i]
    #         P_all[i] = np.linalg.solve(Q, B)

    # --- Step 5: build trajectory function
    bezier_segments = []
    bezier_d1 = []
    bezier_d2 = []

    for i in range(n_seg):
        seg = Bezier([p for p in P_all[i]], t_min=i * seg_time, t_max=(i + 1) * seg_time)
        bezier_segments.append(seg)
        bezier_d1.append(seg.derivative(1))
        bezier_d2.append(seg.derivative(2))

    def trajectory(t):
        if t <= 0:
            return path[0], np.zeros(d), np.zeros(d)
        if t >= total_time:
            return path[-1], np.zeros(d), np.zeros(d)

        i = min(int(t // seg_time), n_seg - 1)
        q = bezier_segments[i](t)
        qdot = bezier_d1[i](t)
        qddot = bezier_d2[i](t)
        return q, qdot, qddot

    return trajectory

import numpy as np
from bezier import Bezier
from scipy.special import comb

def piecewise_bezier_ls(path, total_time, degree=5, lam_smooth=1e-3, n_iter=5, zero_end_vel=True):
    """
    Piecewise Bezier trajectory fitting with least-squares + C1 smoothing + zero end velocities.

    Args:
        path: (N, d) array of key points
        total_time: total duration (s)
        degree: degree of each Bezier segment
        lam_smooth: smoothing weight (second-order diff regularizer)
        n_iter: number of iterative smoothing passes
        zero_end_vel: enforce qdot(0)=qdot(T)=0

    Returns:
        trajectory(t) -> (q, qdot, qddot)
    """
    path = np.asarray(path, dtype=float)
    N, d = path.shape
    n_seg = N - 1
    seg_time = total_time / n_seg

    # Initialize control points: straight line interpolation for each segment
    P_all = np.zeros((n_seg, degree+1, d))
    for i in range(n_seg):
        for k in range(degree+1):
            alpha = k / degree
            P_all[i, k] = (1-alpha)*path[i] + alpha*path[i+1]

    # Second-difference regularizer
    D = np.zeros((degree-1, degree+1))
    for j in range(degree-1):
        D[j, j:j+3] = [1, -2, 1]

    # Iterative least-squares + smoothing
    for it in range(n_iter):
        for i in range(n_seg):
            # Sample path points for this segment
            idx_start = i
            idx_end = i+2  # cover current segment's start/end points
            path_seg = path[idx_start:idx_end]
            t_samples = np.linspace(0, 1, len(path_seg))

            # Bezier basis matrix
            M = np.zeros((len(t_samples), degree+1))
            for k, t in enumerate(t_samples):
                for j in range(degree+1):
                    M[k, j] = comb(degree, j) * (t**j) * ((1-t)**(degree-j))

            # Solve weighted least squares with smoothing
            Q = M.T @ M + lam_smooth * (D.T @ D)
            B = M.T @ path_seg
            P_opt = np.linalg.solve(Q, B)

            # Enforce start/end points
            P_opt[0] = path[i]
            P_opt[-1] = path[i+1]
            if zero_end_vel:
                P_opt[1] = P_opt[0]       # qdot(0)=0
                P_opt[-2] = P_opt[-1]     # qdot(T)=0

            P_all[i] = P_opt

        # --- C1 smoothing at segment junctions ---
        for i in range(1, n_seg):
            v_prev = P_all[i-1][-1] - P_all[i-1][-2]
            v_curr = P_all[i][1] - P_all[i][0]
            avg_v = 0.5*(v_prev + v_curr)
            P_all[i-1][-2] = P_all[i-1][-1] - avg_v
            P_all[i][1] = P_all[i][0] + avg_v

    # Build Bezier segments
    bezier_segments = []
    bezier_d1 = []
    bezier_d2 = []
    for i in range(n_seg):
        seg = Bezier([p for p in P_all[i]], t_min=i*seg_time, t_max=(i+1)*seg_time)
        bezier_segments.append(seg)
        bezier_d1.append(seg.derivative(1))
        bezier_d2.append(seg.derivative(2))

    def trajectory(t):
        if t <= 0:
            return path[0], np.zeros(d), np.zeros(d)
        elif t >= total_time:
            return path[-1], np.zeros(d), np.zeros(d)
        idx = min(int(t // seg_time), n_seg-1)
        q = bezier_segments[idx](t)
        qdot = bezier_d1[idx](t)
        qddot = bezier_d2[idx](t)
        return q, qdot, qddot

    return trajectory


def bezier_QP_trajectory(path, total_time, degree=5, lam_smooth=1e-3):
    """
    QP-based Bezier trajectory fitting.
    path: list of np.array, shape (N, d)
    total_time: total duration
    degree: Bezier degree
    lam_smooth: regularization for smoothness
    """
    path = np.array(path)
    N, d = path.shape
    t_samples = np.linspace(0, 1, N)

    # Step 1. Build Bezier basis matrix M (N x (degree+1))
    M = np.zeros((N, degree + 1))
    for k, t in enumerate(t_samples):
        for i in range(degree + 1):
            M[k, i] = comb(degree, i) * (t ** i) * ((1 - t) ** (degree - i))

    # Step 2. Build smoothness regularizer D (second difference)
    D = np.zeros((degree - 1, degree + 1))
    for i in range(degree - 1):
        D[i, i:i+3] = [1, -2, 1]

    # Step 3. Solve for control points P
    Q = M.T @ M + lam_smooth * (D.T @ D)
    B = M.T @ path

    P_opt = np.linalg.solve(Q, B)  # shape ((degree+1), d)

    # Step 4. Enforce exact start & end points + zero velocities
    P_opt[0] = path[0]
    P_opt[1] = path[0]        # enforce qdot(0)=0
    P_opt[-1] = path[-1]
    P_opt[-2] = path[-1]      # enforce qdot(T)=0

    # Step 5. Build Bezier and derivatives
    bezier_curve = Bezier([p for p in P_opt], t_min=0.0, t_max=total_time)
    bezier_d1 = bezier_curve.derivative(1)
    bezier_d2 = bezier_curve.derivative(2)

    def trajectory(t):
        if t <= 0:
            return path[0], np.zeros(d), np.zeros(d)
        elif t >= total_time:
            return path[-1], np.zeros(d), np.zeros(d)
        q = bezier_curve(t)
        qdot = bezier_d1(t)
        qddot = bezier_d2(t)
        return q, qdot, qddot

    return trajectory



def interpolate_path_t(path, total_time):
    # path: [q0, q1, q2, ..., qN]
    # path = [np.array(q, dtype=float) for q in path]
    N = len(path) - 1
    segment_time = total_time / N

    def trajectory(t):
        if t <= 0: return path[0], np.zeros_like(path[0]), np.zeros_like(path[0])
        if t >= total_time: return path[-1], np.zeros_like(path[0]), np.zeros_like(path[0])

        idx = int(t // segment_time)
        tau = (t % segment_time) / segment_time
        q = (1 - tau) * path[idx] + tau * path[idx + 1]
        qdot = (path[idx + 1] - path[idx]) / segment_time
        qddot = np.zeros_like(q)
        return q, qdot, qddot

    return trajectory

import numpy as np

def interpolate_path(path, total_time):
    # path: list of np.ndarray, e.g. [q0, q1, q2, ..., qN]
    path = [np.array(p, dtype=float) for p in path]
    N = len(path) - 1

    # ---- 1. 计算每段长度（欧氏距离）----
    seg_lengths = [np.linalg.norm(path[i+1] - path[i]) for i in range(N)]
    total_length = sum(seg_lengths)

    # ---- 2. 按比例分配每段时间 ----
    seg_times = [total_time * (l / total_length) for l in seg_lengths]

    # ---- 3. 累积时间表 ----
    cum_times = np.concatenate(([0], np.cumsum(seg_times)))

    # ---- 4. 返回一个 trajectory(t) 函数 ----
    def trajectory(t):
        # 边界
        if t <= 0:
            return path[0], np.zeros_like(path[0]), np.zeros_like(path[0])
        if t >= total_time:
            return path[-1], np.zeros_like(path[0]), np.zeros_like(path[0])

        # 找到当前所在段
        idx = np.searchsorted(cum_times, t, side='right') - 1

        t0, t1 = cum_times[idx], cum_times[idx+1]
        tau = (t - t0) / (t1 - t0)

        # ---- 匀速线性插值 ----
        q0, q1 = path[idx], path[idx + 1]
        q = (1 - tau) * q0 + tau * q1
        qdot = (q1 - q0) / (t1 - t0)
        qddot = np.zeros_like(q)  # 匀速，无加速度

        return q, qdot, qddot

    return trajectory

# def interpolate_path_zero_end_vel(path, total_time):
#     """
#     分段线性插值轨迹 + 首末速度为0
#     path: list of np.ndarray [q0, q1, ..., qN]
#     total_time: 总运动时间
#     Returns: trajectory(t) -> (q, qdot, qddot)
#     """
#     path = [np.array(p, dtype=float) for p in path]
#     N = len(path) - 1

#     # 1. 每段长度 & 时间分配
#     seg_lengths = [np.linalg.norm(path[i+1] - path[i]) for i in range(N)]
#     total_length = sum(seg_lengths)
#     seg_times = [total_time * (l / total_length) for l in seg_lengths]
#     cum_times = np.concatenate(([0], np.cumsum(seg_times)))

#     # 2. 平滑速度修正函数：缓启动缓停（3次多项式）
#     def smooth_factor(s):
#         # s in [0,1] -> factor from 0->1->0
#         # 3次多项式: 3s^2 - 2s^3
#         return 3*s**2 - 2*s**3

#     def trajectory(t):
#         # 边界
#         if t <= 0:
#             return path[0], np.zeros_like(path[0]), np.zeros_like(path[0])
#         if t >= total_time:
#             return path[-1], np.zeros_like(path[0]), np.zeros_like(path[0])

#         # 找到当前段
#         idx = min(np.searchsorted(cum_times, t, side='right') - 1, N-1)
#         t0, t1 = cum_times[idx], cum_times[idx+1]
#         tau = (t - t0) / (t1 - t0)

#         # 线性插值
#         q0, q1 = path[idx], path[idx+1]
#         delta = q1 - q0

#         # 缓启动/缓停修正：首段加启动，末段加停
#         factor = tau
#         if idx == 0:  # 第一段
#             factor = smooth_factor(tau)
#         elif idx == N-1:  # 最后一段
#             factor = smooth_factor(tau)  # 可反向也行
#         # 其他段可以选 linear 或 smooth_factor(tau) 保持平滑

#         q = q0 + factor * delta
#         # 速度 = d(q)/dt
#         qdot = delta / (t1 - t0) * (6*tau*(1-tau))  # 3s^2-2s^3 的导数 6s(1-s)
#         qddot = delta / (t1 - t0)**2 * (6 - 12*tau)  # 二阶导

#         return q, qdot, qddot

#     return trajectory


def interpolate_path_zero_end_vel(path, total_time):
    """
    分段线性插值轨迹 + 整体首尾速度为0
    中间段速度匀速
    path: list of np.ndarray [q0, q1, ..., qN]
    total_time: 总运动时间
    Returns: trajectory(t) -> (q, qdot, qddot)
    """
    path = [np.array(p, dtype=float) for p in path]
    N = len(path) - 1

    # 每段长度 & 时间分配
    seg_lengths = [np.linalg.norm(path[i+1] - path[i]) for i in range(N)]
    total_length = sum(seg_lengths)
    seg_times = [total_time * (l / total_length) for l in seg_lengths]
    cum_times = np.concatenate(([0], np.cumsum(seg_times)))

    # 缓启动/缓停函数
    def smooth_factor(s):
        return 3*s**2 - 2*s**3

    def trajectory(t):
        # 边界
        if t <= 0:
            return path[0], np.zeros_like(path[0]), np.zeros_like(path[0])
        if t >= total_time:
            return path[-1], np.zeros_like(path[0]), np.zeros_like(path[0])

        # 找到当前段
        idx = min(np.searchsorted(cum_times, t, side='right') - 1, N-1)
        t0, t1 = cum_times[idx], cum_times[idx+1]
        tau = (t - t0) / (t1 - t0)
        q0, q1 = path[idx], path[idx+1]
        delta = q1 - q0

        # 首尾段用 smooth_factor, 中间段匀速
        if idx == 0:  # 第一段
            factor = smooth_factor(tau)
            qdot = delta / (t1 - t0) * (6*tau*(1-tau))
            qddot = delta / (t1 - t0)**2 * (6 - 12*tau)
        elif idx == N-1:  # 最后一段
            factor = smooth_factor(tau)
            qdot = delta / (t1 - t0) * (6*tau*(1-tau))
            qddot = delta / (t1 - t0)**2 * (6 - 12*tau)
        else:  # 中间段匀速
            factor = tau
            qdot = delta / (t1 - t0)
            qddot = np.zeros_like(delta)

        q = q0 + factor * delta
        return q, qdot, qddot

    return trajectory


if __name__ == "__main__":

    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from setup_meshcat import updatevisuals
    from path import computepath
    import time
    
    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6000")
    q0 = robot.q0.copy()

    # discretisationsteps = 10
    # discretisationdist = 0.01
    # k = 100
        
    # from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    # robot, sim, cube = setupwithpybullet()
    
    
    # from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    # from inverse_geometry import computeqgrasppose
    

    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    
    #setting initial configuration
    # sim.setqsim(q0)
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities
    # def maketraj(q0,q1,T): #TODO compute a real trajectory !
    #     q_of_t = Bezier([q0,q0,q1,q1],t_max=T)
    #     vq_of_t = q_of_t.derivative(1)
    #     vvq_of_t = vq_of_t.derivative(1)
    #     return q_of_t, vq_of_t, vvq_of_t
    
    
    #TODO this is just a random trajectory, you need to do this yourself
    total_time=4.
    # traj = interpolate_path(path, total_time)   #bezier_trajectory
    traj = piecewise_bezier_trajectory(path, total_time)

    print("path len: ", len(path))
    if len(path) > 0:
        for q in path:
            viz.display(q)
            time.sleep(1)

    # display the traj    
    tcur = 0.
    while tcur < total_time:
        q, qd, qdd = traj(tcur)
        viz.display(q)
    #     rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT