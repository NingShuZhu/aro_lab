import numpy as np
from bezier import Bezier

from scipy.special import comb

def bezier_QP_trajectory_improved(path, total_time, degree=7, lam_smooth=1e-4, discretisation_t_by_arclength=True,
                                  weights=None, force_through_indices=None):
    # QP-fitting Bezier trajectory:
    path = np.asarray(path, dtype=float)
    N, d = path.shape
    if degree < 1:
        raise ValueError("degree must be >=1")
    if N < 2:
        raise ValueError("need at least two path points")

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

    M = np.zeros((N, degree + 1))
    for k, u in enumerate(t_samples):
        for i in range(degree + 1):
            M[k, i] = comb(degree, i) * (u ** i) * ((1 - u) ** (degree - i))

    # smoothing regularizer
    if degree >= 2:
        D = np.zeros((degree - 1, degree + 1))
        for i in range(degree - 1):
            D[i, i:i+3] = [1.0, -2.0, 1.0]
    else:
        D = np.zeros((0, degree + 1))

    if weights is None:
        W = np.eye(N)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        assert w.shape[0] == N
        W = np.diag(w)

    # Solve (M^T W M + lam * D^T D) P = M^T W X
    A = M.T @ W @ M
    if D.shape[0] > 0 and lam_smooth > 0:
        A += lam_smooth * (D.T @ D)
    B = M.T @ W @ path  # shape (degree+1, d)

    P = None
    if not force_through_indices:
        P = np.linalg.solve(A, B)
    else:
        C = np.array(sorted(force_through_indices), dtype=int)
        # Build constraint matrix for P
        C_mat = M[C, :]
        Xc = path[C, :]

        P_part_flat = np.linalg.pinv(C_mat) @ Xc 

        # Use SVD to get nullspace
        U, S, Vt = np.linalg.svd(C_mat, full_matrices=True)
        rank = np.sum(S > 1e-12)
        nullspace = Vt[rank:].T
        if nullspace.size == 0:
            # no nullspace, P is fully constrained by C_mat
            P = P_part_flat
        else:
            AN = A @ nullspace
            rhs = B - A @ P_part_flat
            # Solve z (least-squares)
            z, *_ = np.linalg.lstsq(AN, rhs, rcond=None)
            P = P_part_flat + nullspace @ z

    # enforce endpoints
    P[0, :] = path[0, :]
    P[-1, :] = path[-1, :]
    # Enforce zero velocities at endpoints by duplication:
    P[1, :] = P[0, :].copy()
    P[-2, :] = P[-1, :].copy()

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


def piecewise_bezier_trajectory(path, total_time, degree=5, lam_smooth=1e-4, continuity='C1', zero_end_vel=True):
    
    path = np.asarray(path, dtype=float)
    N, d = path.shape
    n_seg = N - 1
    seg_time = total_time / n_seg

    # each segment has degree+1 control points
    P_all = np.zeros((n_seg, degree + 1, d))

    # 1: initial guess: straight line control points
    for i in range(n_seg):
        for k in range(degree + 1):
            alpha = k / degree
            P_all[i, k] = (1 - alpha) * path[i] + alpha * path[i + 1]

    # 2: enforce continuity constraints

    # Smoothness regularization between adjacent segments
    for i in range(1, n_seg):
        # continuity constraints
        if continuity in ['C1', 'C2']:
            # derivative continuity at junction
            # qdot at end of segment i-1 = qdot at start of segment i
            P_prev = P_all[i - 1]
            P_curr = P_all[i]
            # derivative for Bezier
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

    # 3: zero end velocities
    if zero_end_vel:
        P_all[0, 1] = P_all[0, 0]  # qdot(0)=0
        P_all[-1, -2] = P_all[-1, -1]  # qdot(T)=0

    # 4: smooth within each segment (optional lam_smooth)
    # if lam_smooth > 0:
    #     for i in range(n_seg):
    #         D = np.zeros((degree - 1, degree + 1))
    #         for j in range(degree - 1):
    #             D[j, j:j+3] = [1, -2, 1]
    #         Q = np.eye(degree + 1) + lam_smooth * (D.T @ D)
    #         B = P_all[i]
    #         P_all[i] = np.linalg.solve(Q, B)

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


# QP-based Bezier trajectory fitting
def bezier_QP_trajectory(path, total_time, degree=5, lam_smooth=1e-3):
    path = np.array(path)
    N, d = path.shape
    t_samples = np.linspace(0, 1, N)

    # build Bezier basis matrix M 
    M = np.zeros((N, degree + 1))
    for k, t in enumerate(t_samples):
        for i in range(degree + 1):
            M[k, i] = comb(degree, i) * (t ** i) * ((1 - t) ** (degree - i))

    # build smoothness regularizer D (second difference)
    D = np.zeros((degree - 1, degree + 1))
    for i in range(degree - 1):
        D[i, i:i+3] = [1, -2, 1]

    # solve for control points P
    Q = M.T @ M + lam_smooth * (D.T @ D)
    B = M.T @ path

    P_opt = np.linalg.solve(Q, B)

    # enforce exact start and end points, zero velocities
    P_opt[0] = path[0]
    P_opt[1] = path[0]        # enforce qdot(0)=0
    P_opt[-1] = path[-1]
    P_opt[-2] = path[-1]      # enforce qdot(T)=0

    # build Bezier and derivatives
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


# simple interpolation, each segment in the path is allocated the same time
def interpolate_path_t(path, total_time):
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


# interpolating the path: uniform speed for the whole trajectory
def interpolate_path(path, total_time):
    path = [np.array(p, dtype=float) for p in path]
    N = len(path) - 1

    # calculate the distance between each pair
    seg_lengths = [np.linalg.norm(path[i+1] - path[i]) for i in range(N)]
    total_length = sum(seg_lengths)

    # allocate time according to the distance
    seg_times = [total_time * (l / total_length) for l in seg_lengths]

    cum_times = np.concatenate(([0], np.cumsum(seg_times)))

    def trajectory(t):
        if t <= 0:
            return path[0], np.zeros_like(path[0]), np.zeros_like(path[0])
        if t >= total_time:
            return path[-1], np.zeros_like(path[0]), np.zeros_like(path[0])

        idx = np.searchsorted(cum_times, t, side='right') - 1

        t0, t1 = cum_times[idx], cum_times[idx+1]
        tau = (t - t0) / (t1 - t0)

        # uniform linear interpolation
        q0, q1 = path[idx], path[idx + 1]
        q = (1 - tau) * q0 + tau * q1
        qdot = (q1 - q0) / (t1 - t0)
        qddot = np.zeros_like(q)  # no acceleration

        return q, qdot, qddot

    return trajectory


# start and end velocity = 0, uniform velocity for the intermediate sections
def interpolate_path_zero_end_vel(path, total_time):
    path = [np.array(p, dtype=float) for p in path]
    N = len(path) - 1

    seg_lengths = [np.linalg.norm(path[i+1] - path[i]) for i in range(N)]
    total_length = sum(seg_lengths)
    seg_times = [total_time * (l / total_length) for l in seg_lengths]
    cum_times = np.concatenate(([0], np.cumsum(seg_times)))

    # function for the start/end section
    def smooth_factor(s):
        return 3*s**2 - 2*s**3

    def trajectory(t):
        if t <= 0:
            return path[0], np.zeros_like(path[0]), np.zeros_like(path[0])
        if t >= total_time:
            return path[-1], np.zeros_like(path[0]), np.zeros_like(path[0])

        idx = min(np.searchsorted(cum_times, t, side='right') - 1, N-1)
        t0, t1 = cum_times[idx], cum_times[idx+1]
        tau = (t - t0) / (t1 - t0)
        q0, q1 = path[idx], path[idx+1]
        delta = q1 - q0

        if idx == 0:  # start section
            factor = smooth_factor(tau)
            qdot = delta / (t1 - t0) * (6*tau*(1-tau))
            qddot = delta / (t1 - t0)**2 * (6 - 12*tau)
        elif idx == N-1:  # end section
            factor = smooth_factor(tau)
            qdot = delta / (t1 - t0) * (6*tau*(1-tau))
            qddot = delta / (t1 - t0)**2 * (6 - 12*tau)
        else:  # intermediate sections
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

    from config import DT

    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    total_time=4.
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
        tcur += DT