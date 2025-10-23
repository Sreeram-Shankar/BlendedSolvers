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

#defines the bdf4 step
def step_bdf4(f, t_n, Y_hist, h):
    y_n, y_nm1, y_nm2, y_nm3 = Y_hist
    t_np1 = t_n + h

    def residual(y_next):
        return ((25/12)*y_next - 4*y_n + 3*y_nm1 - (4/3)*y_nm2 + 0.25*y_nm3)/h - f(t_np1, y_next)

    def jac(y_next):
        Jf = finite_diff_jac(lambda z: f(t_np1, z), y_next)
        n = len(y_n)
        return (25/(12*h))*np.eye(n) - Jf

    y0 = y_n.copy()
    return newton_solve(residual, y0, jac)

#defines the lobatto3 coefficients
A_lobatto3c = np.array([
    [0,          0,          0],
    [5/24,       1/3,       -1/24],
    [1/6,        2/3,        1/6]
])
b_lobatto3c = np.array([1/6, 2/3, 1/6])
c_lobatto3c = np.array([0, 1/2, 1])

#defines the lobatto3 step
def step_lobatto3c(f, t, y, h, y_guess=None):
    s, n = 3, len(y)
    if y_guess is None:
        y_guess = y.copy()
    Y = np.zeros((s, n))

    def residual(Z):
        Z = Z.reshape((s, n))
        R = np.zeros_like(Z)
        for i in range(s):
            acc = np.zeros_like(y)
            for j in range(s):
                acc += A_lobatto3c[i,j] * f(t + c_lobatto3c[j]*h, Z[j])
            R[i] = Z[i] - y - h*acc
        return R.ravel()

    def jac(Z):
        Z = Z.reshape((s, n))
        Jtot = np.zeros((s*n, s*n))
        for i in range(s):
            for j in range(s):
                Jf = finite_diff_jac(lambda z: f(t + c_lobatto3c[j]*h, z), Z[j])
                block = np.eye(n)*(1.0 if i==j else 0.0) - h*A_lobatto3c[i,j]*Jf
                Jtot[i*n:(i+1)*n, j*n:(j+1)*n] = block
        return Jtot

    Z0 = np.tile(y_guess, s)
    Z = newton_solve(residual, Z0, jac)
    Z = Z.reshape((s, n))
    K = np.zeros_like(Z)
    for i in range(s):
        K[i] = f(t + c_lobatto3c[i]*h, Z[i])
    y_next = y + h * np.sum(b_lobatto3c[i] * K[i] for i in range(s))
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


#main solver for blended4
def solve_blended4(f, t_span, y0, h, p=1.5, a_min=0.05, a_max=0.98):
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0

    #3 rk4 steps for bootstrap
    def rk4_step(f, t, y, h):
        k1 = f(t, y)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    for k in range(3): Y[k+1] = rk4_step(f, t_grid[k], Y[k], h)


    f_prev = f(t_grid[3], Y[3])
    a_hist = [0.0]*4

    for n in range(3, N):
        t = t_grid[n]
        y_hist = [Y[n], Y[n-1], Y[n-2], Y[n-3]]
        f_n = f(t, Y[n])
        sigma = stiffness_proxy(h, f_n, f_prev, Y[n], Y[n-1])
        a = adapt_a(sigma, p, a_min, a_max)

        y_lobatto = step_lobatto3c(f, t, Y[n], h)
        y_bdf4 = step_bdf4(f, t, y_hist, h)

        y_next = a * y_bdf4 + (1 - a) * y_lobatto
        Y[n+1] = y_next
        f_prev = f_n
        a_hist.append(a)

    return t_grid, Y, np.array(a_hist)
