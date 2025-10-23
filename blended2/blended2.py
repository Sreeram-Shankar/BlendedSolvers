import numpy as np

#defines the finite difference approximation of the Jacobian
def finite_diff_jac(fun, x, eps=1e-8):
    n = len(x)
    f0 = fun(x)
    J = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    return J

#solve the nonlinear system of equations
def newton_solve(residual, y0, jac=None, tol=1e-10, max_iter=12):
    y = y0.copy()
    for _ in range(max_iter):
        r = residual(y)
        if np.linalg.norm(r) < tol:
            return y
        J = jac(y) if jac else finite_diff_jac(residual, y)
        dy = np.linalg.solve(J, -r)
        y += dy
        if np.linalg.norm(dy) < tol:
            break
    return y

#defines the guass-legendre1 step
def step_gl1(f, t, y, h, y_guess=None):
    t_mid = t + 0.5 * h

    def residual(y_next):
        y_mid = 0.5 * (y + y_next)
        return y_next - y - h * f(t_mid, y_mid)

    def jac(y_next):
        y_mid = 0.5 * (y + y_next)
        Jf = finite_diff_jac(lambda z: f(t_mid, z), y_mid)
        return np.eye(len(y)) - 0.5 * h * Jf

    y0 = y_guess if y_guess is not None else y.copy()
    return newton_solve(residual, y0, jac)

#defines the bdf2 step
def step_bdf2(f, t, y, y_prev, h, y_guess=None):
    t_next = t + h

    def residual(y_next):
        return (3 * y_next - 4 * y + y_prev) / (2 * h) - f(t_next, y_next)

    def jac(y_next):
        Jf = finite_diff_jac(lambda z: f(t_next, z), y_next)
        return (3 / (2 * h)) * np.eye(len(y)) - Jf

    y0 = y_guess if y_guess is not None else y.copy()
    return newton_solve(residual, y0, jac)

#measures the stiffness of the problem
def stiffness_proxy(h, f_n, f_prev, y_n, y_prev, eps=1e-14):
    num = np.linalg.norm(f_n - f_prev)
    den = max(np.linalg.norm(y_n - y_prev), eps)
    return h * num / den

#defines the adaptive parameter a
def adapt_a(sigma, p=1.5, a_min=0.05, a_max=0.98):
    a = (sigma**p) / (1 + sigma**p)
    return float(np.clip(a, a_min, a_max))

#main solver for blended2
def solve_blended(f, t_span, y0, h, p=1.5, a_min=0.05, a_max=0.98):
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, tf, N + 1)
    Y = np.zeros((N + 1, len(y0)))
    Y[0] = y0

    f_prev = f(t0, y0)
    y1 = step_gl1(f, t0, y0, h)
    Y[1] = y1

    a_hist = [0.0, 0.0]

    for n in range(1, N):
        t = t_grid[n]
        y = Y[n]
        y_prev = Y[n - 1]
        f_n = f(t, y)
        sigma = stiffness_proxy(h, f_n, f_prev, y, y_prev)
        a = adapt_a(sigma, p, a_min, a_max)

        y_gl1 = step_gl1(f, t, y, h, y_guess=y)
        y_bdf2 = step_bdf2(f, t, y, y_prev, h, y_guess=y_gl1)

        y_next = a * y_bdf2 + (1 - a) * y_gl1
        Y[n + 1] = y_next

        a_hist.append(a)
        f_prev = f_n

    return t_grid, Y, np.array(a_hist)
