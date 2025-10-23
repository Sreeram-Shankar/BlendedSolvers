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

#solves the nonlinear system of equations
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


#defines the trbdf2 step
def step_trbdf2(f, t, y, h, alpha=2 - np.sqrt(2)):
    #trapezoidal rule predictor
    t1 = t + alpha * h / 2.0

    def residual_stage1(y1): return y1 - y - alpha*h * 0.5 * (f(t, y) + f(t1, y1))

    def jac_stage1(y1):
        Jf = finite_diff_jac(lambda z: f(t1, z), y1)
        n = len(y)
        return np.eye(n) - 0.5 * alpha * h * Jf

    y1 = newton_solve(residual_stage1, y.copy(), jac_stage1)

    #bdf2 corrector
    t2 = t + h

    def residual_stage2(y2): return ((1/(2 - alpha)) * ((1 - alpha)**2 * y + alpha**2 * y1) + ((1 - alpha)*alpha*h / (2 - alpha)) * f(t2, y2) - y2)

    def jac_stage2(y2):
        Jf = finite_diff_jac(lambda z: f(t2, z), y2)
        n = len(y)
        coef = (1 - alpha)*alpha*h / (2 - alpha)
        return np.eye(n) - coef * Jf

    y2 = newton_solve(residual_stage2, y1.copy(), jac_stage2)
    return y2



#defines the chebyshev2 step as an implicit prototype method
def step_chebyshev2(f, t, y, h):
    a21 = 0.5
    a22 = 0.5
    b1, b2 = 0.5, 0.5
    c2 = 0.5

    n = len(y)
    k1 = f(t, y)

    def residual(y2): return y2 - y - h*(a21*k1 + a22*f(t + c2*h, y2))

    def jac(y2):
        Jf = finite_diff_jac(lambda z: f(t + c2*h, z), y2)
        return np.eye(n) - h*a22*Jf

    y2 = newton_solve(residual, y.copy(), jac)
    k2 = f(t + c2*h, y2)
    y_next = y + h*(b1*k1 + b2*k2)
    return y_next

#measures the stiffness of the problem
def stiffness_proxy(h, f_n, f_prev, y_n, y_prev, eps=1e-14):
    num = np.linalg.norm(f_n - f_prev)
    den = max(np.linalg.norm(y_n - y_prev), eps)
    return h * num / den

#defines the adaptive parameter a
def adapt_a(sigma, p=1.5, a_min=0.05, a_max=0.98):
    a = (sigma**p) / (1 + sigma**p)
    return float(np.clip(a, a_min, a_max))


#main solver for blendedTRBDFC
def solve_blended2plus(f, t_span, y0, h, p=1.5, a_min=0.05, a_max=0.98):
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0

    #single step rk2 for bootstrap
    def rk2_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + h, y + h*k1)
        return y + 0.5*h*(k1 + k2)

    Y[1] = rk2_step(f, t_grid[0], y0, h)
    f_prev = f(t_grid[1], Y[1])
    a_hist = [0.0, 0.0]

    for n in range(1, N):
        t = t_grid[n]
        y = Y[n]
        y_prev = Y[n-1]
        f_n = f(t, y)

        sigma = stiffness_proxy(h, f_n, f_prev, y, y_prev)
        a = adapt_a(sigma, p, a_min, a_max)

        # Compute both implicit methods
        y_cheb = step_chebyshev2(f, t, y, h)
        y_trbdf2 = step_trbdf2(f, t, y, h)

        y_next = (1 - a)*y_cheb + a*y_trbdf2
        Y[n+1] = y_next
        a_hist.append(a)
        f_prev = f_n

    return t_grid, Y, np.array(a_hist)
