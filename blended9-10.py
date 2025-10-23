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

#load the butcher coefficients
def load_tableau(Afile, bfile, cfile):
    A = np.loadtxt(Afile)
    b = np.loadtxt(bfile)
    c = np.loadtxt(cfile)
    return A, b, c

GAUSS_A, GAUSS_b, GAUSS_c = load_tableau("gl5/gauss_legendre_s5_A.txt", "gl5/gauss_legendre_s5_b.txt", "gl5/gauss_legendre_s5_c.txt")
RADAU_A, RADAU_b, RADAU_c = load_tableau("radau5/radau_s5_A.txt", "radau5/radau_s5_b.txt", "radau5/radau_s5_c.txt")

#generic implicit RK step
def step_irk(f, t, y, h, A, b, c, y_guess=None, tol=1e-10, max_iter=10):
    s = len(b)
    n = len(y)
    Ystages = np.zeros((s, n))
    if y_guess is None:
        y_guess = y.copy()
    K = np.zeros((s, n))

    def residual(Z):
        Z = Z.reshape((s, n))
        R = np.zeros_like(Z)
        for i in range(s):
            ti = t + c[i] * h
            yi = y + h * np.sum(A[i, j] * f(t + c[j] * h, Z[j]) for j in range(s))
            R[i] = Z[i] - yi
        return R.ravel()

    def jac(Z):
        Z = Z.reshape((s, n))
        Jtot = np.zeros((s * n, s * n))
        for i in range(s):
            for j in range(s):
                Jf = finite_diff_jac(lambda z: f(t + c[j] * h, z), Z[j])
                block = np.eye(n) * (1.0 if i == j else 0.0) - h * A[i, j] * Jf
                Jtot[i*n:(i+1)*n, j*n:(j+1)*n] = block
        return Jtot

    Z0 = np.tile(y_guess, s)
    Z = newton_solve(residual, Z0, jac=jac, tol=tol, max_iter=max_iter)
    Z = Z.reshape((s, n))
    for i in range(s):
        K[i] = f(t + c[i] * h, Z[i])
    y_next = y + h * np.sum(b[i] * K[i] for i in range(s))
    return y_next

#measures the stiffness of the problem
def stiffness_proxy(h, f_n, f_prev, y_n, y_prev, eps=1e-14):
    num = np.linalg.norm(f_n - f_prev)
    den = max(np.linalg.norm(y_n - y_prev), eps)
    return h * num / den

#defines the adaptive parameter a
def adapt_a(sigma, p=1.5, a_min=0.05, a_max=0.98):
    a = (sigma ** p) / (1 + sigma ** p)
    return float(np.clip(a, a_min, a_max))

#main solver for radau5 and guass-legendre5 in blended9-10
def solve_blended9_10(f, t_span, y0, h, p=1.5, a_min=0.05, a_max=0.98):
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, tf, N + 1)
    Y = np.zeros((N + 1, len(y0)))
    Y[0] = y0
    f_prev = f(t0, y0)

    #single Guass step for bootstrap
    y1 = step_irk(f, t0, y0, h, GAUSS_A, GAUSS_b, GAUSS_c)
    Y[1] = y1
    a_hist = [0.0, 0.0]

    for n in range(1, N):
        t = t_grid[n]
        y = Y[n]
        y_prev = Y[n - 1]
        f_n = f(t, y)

        sigma = stiffness_proxy(h, f_n, f_prev, y, y_prev)
        a = adapt_a(sigma, p, a_min, a_max)

        y_gauss = step_irk(f, t, y, h, GAUSS_A, GAUSS_b, GAUSS_c)
        y_radau = step_irk(f, t, y, h, RADAU_A, RADAU_b, RADAU_c, y_guess=y_gauss)
        y_next = a * y_radau + (1 - a) * y_gauss

        Y[n + 1] = y_next
        a_hist.append(a)
        f_prev = f_n

    return t_grid, Y, np.array(a_hist)
