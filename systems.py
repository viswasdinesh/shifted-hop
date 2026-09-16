"""Toy systems extracted from HOP_colab_notebook.ipynb."""

import numpy as np

def make_double_integrator(dt=0.05, N=120):
    def F(x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.array([x[0] + dt * x[1], x[1] + dt * u[0]])
    F.dt = dt
    x0    = np.array([1.0, 0.0])
    xg    = np.array([2.0, 0.0])
    u_ref = np.array([0.0])
    Q     = np.diag([1.0, 0.1])
    R     = np.array([[1e-2]])
    alpha = 50.0
    w     = 0.02
    T_min, T_max = 10, 80
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max

def make_cartpole(dt=0.05, N=120, b_p=0.1, b_th=0.05):
    M, m, l, g = 1.0, 0.1, 0.5, 9.81

    def F(x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)) or float(np.linalg.norm(x)) > 1e6:
            return np.full(4, np.nan)
        p, th, pdot, thdot = x
        f = u[0]
        s, c = np.sin(th), np.cos(th)
        denom = M + m * s**2
        pddot  = (f + m*s*(l*thdot**2 + g*c)) / denom - b_p*pdot
        thddot = (-f*c - m*l*thdot**2*c*s - (M+m)*g*s) / (l*denom) - b_th*thdot
        xdot = np.array([pdot, thdot, pddot, thddot])
        return x + dt * xdot

    F.dt = dt
    x0    = np.array([0.0, np.pi, 0.0, 0.0])   # cart at origin, pole hanging down
    xg    = np.array([0.0, 0.0, 0.0, 0.0])     # cart at origin, pole upright
    u_ref = np.array([0.0])
    Q     = np.diag([1.0, 10.0, 0.1, 0.1])
    R     = np.array([[0.01]])
    alpha = 150.0
    w     = 0.05
    T_min, T_max = 40, 120
    wrap_idx = [1]
    return F, x0, xg, u_ref, Q, R, alpha, w, N, T_min, T_max, wrap_idx


def make_quadrotor(dt=0.05, N=160):
    mass, g = 1.0, 9.81
    Ix, Iy, Iz = 0.02, 0.02, 0.04
    kv, kw = 0.05, 0.01
    I_body = np.diag([Ix, Iy, Iz])
    I_inv  = np.diag([1/Ix, 1/Iy, 1/Iz])

    def rotm(phi, th, psi):
        s, c = np.sin, np.cos
        Rz = np.array([[c(psi),-s(psi),0],[s(psi),c(psi),0],[0,0,1.0]])
        Ry = np.array([[c(th),0,s(th)],[0,1,0],[-s(th),0,c(th)]])
        Rx = np.array([[1,0,0],[0,c(phi),-s(phi)],[0,s(phi),c(phi)]])
        return Rz @ Ry @ Rx

    def Tmat(phi, th):
        s, c, t = np.sin, np.cos, np.tan
        return np.array([[1, s(phi)*t(th), c(phi)*t(th)],
                         [0, c(phi), -s(phi)],
                         [0, s(phi)/c(th), c(phi)/c(th)]])

    def F(x, u):
        x = np.asarray(x, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
            return np.full(12, np.nan)
        if float(np.linalg.norm(x)) > 1e6:
            return np.full(12, np.nan)
        phi, th, psi = x[6:9]
        if abs(np.cos(th)) < 1e-3:
            return np.full(12, np.nan)
        vel, omg = x[3:6], x[9:12]
        Rb = rotm(phi, th, psi)
        e3 = np.array([0, 0, 1.0])
        acc = Rb @ (e3 * u[0]) / mass - np.array([0, 0, g]) - kv * vel
        eulerdot = Tmat(phi, th) @ omg
        omgdot = I_inv @ (u[1:4] - np.cross(omg, I_body @ omg)) - kw * omg
        xdot = np.zeros(12)
        xdot[0:3] = vel; xdot[3:6] = acc
        xdot[6:9] = eulerdot; xdot[9:12] = omgdot
        return x + dt * xdot

    F.dt = dt
    x0 = np.zeros(12); x0[:3] = [2, 2, 2]
    xg = np.zeros(12)
    u_ref = np.array([mass * g, 0, 0, 0])
    Q_q = np.diag([5,5,5, 1,1,1, 20,20,10, 1,1,1]).astype(float)
    R_q = np.diag([1e-3, 1e-2, 1e-2, 1e-2]).astype(float)
    return F, x0, xg, u_ref, Q_q, R_q, 300.0, 0.05, N, 20, 160, [6,7,8]
