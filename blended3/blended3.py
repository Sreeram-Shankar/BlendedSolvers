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


#defines the sdirk(4,3) coefficients
gamma = 0.435866521508458999416019  
A_sdirk = np.array([
    [gamma, 0, 0, 0],
    [0.5 - gamma, gamma, 0, 0],
    [2*gamma, 1 - 4*gamma, gamma, 0],
    [(-3*gamma + 1), (2*gamma - 1), gamma, gamma]
])
b_sdirk = np.array([(-3*gamma + 1), (2*gamma - 1), gamma, gamma])
c_sdirk = np.sum(A_sdirk, axis=1)

#defines the sdirk(4,3) step
def step_sdirk43(f, t, y, h, y_guess=None):
    s, n = len(b_sdirk), len(y)
    if y_guess is None:
        y_guess = y.copy()
    Y = np.zeros((s, n))
    for i in range(s):
        ti = t + c_sdirk[i]*h

        def residual(Yi):
            acc = np.zeros_like(y)
            for j in range(i+1):  # DIRK structure
                Yj = Y[j] if j < i else Yi
                acc += A_sdirk[i,j] * f(t + c_sdirk[j]*h, Yj)
            return Yi - y - h*acc

        def jac(Yi):
            Jf = finite_diff_jac(lambda z: f(ti, z), Yi)
            return np.eye(n) - h*A_sdirk[i,i]*Jf

        Yi0 = y_guess if i == 0 else Y[i-1]
        Y[i] = newton_solve(residual, Yi0, jac)
    y_next = y + h * np.sum(b_sdirk[i] * f(t + c_sdirk[i]*h, Y[i]) for i in range(s))
    return y_next


#defines the adams-moulton3 step
def step_am3(f, t_n, y_n, y_nm1, h, f_n=None):
    if f_n is None:
        f_n = f(t_n, y_n)
    t_np1 = t_n + h

    def residual(y_next):
        f_np1 = f(t_np1, y_next)
        return y_next - y_n - (h/12.0)*(5*f_np1 + 8*f_n - f(t_n - h, y_nm1))

    def jac(y_next):
        Jf = finite_diff_jac(lambda z: f(t_np1, z), y_next)
        n = len(y_n)
        return np.eye(n) - (5*h/12.0)*Jf

    y0 = y_n.copy()
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


#main solver for blended3
def solve_blended3(f, t_span, y0, h, p=1.5, a_min=0.05, a_max=0.98):
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, tf, N + 1)
    Y = np.zeros((N + 1, len(y0)))
    Y[0] = y0
    f_prev = f(t0, y0)

    #single step euler for bootstrap
    y1 = y0 + h * f_prev
    Y[1] = y1
    a_hist = [0.0, 0.0]

    for n in range(1, N):
        t = t_grid[n]
        y = Y[n]
        y_prev = Y[n - 1]
        f_n = f(t, y)

        sigma = stiffness_proxy(h, f_n, f_prev, y, y_prev)
        a = adapt_a(sigma, p, a_min, a_max)

        y_am3 = step_am3(f, t, y, y_prev, h, f_n)
        y_sdirk = step_sdirk43(f, t, y, h, y_guess=y_am3)

        y_next = a * y_sdirk + (1 - a) * y_am3
        Y[n + 1] = y_next
        a_hist.append(a)
        f_prev = f_n

    return t_grid, Y, np.array(a_hist)
