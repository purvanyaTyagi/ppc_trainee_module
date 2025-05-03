import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.degree = len(coeffs) - 1
    def __call__(self, x):
        val = 0
        for i in range(self.degree + 1):
            power = self.degree - i
            val += self.coeffs[i] * ((x) ** power)
        return val

    def derivative(self):
        coeffs_derv = []
        for i in range(self.degree):
            power = self.degree - i
            coeffs_derv.append(power * self.coeffs[i])
        return polynomial(coeffs_derv)
    
    def __print__(self):
        print("Polynomial coefficients: ", self.coeffs)

def gaussian_elimination(A, B):
    n = len(B)
    # Forward Elimination
    for i in range(n):
        # Pivot
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]
        B[i], B[max_row] = B[max_row], B[i]

        # Eliminate
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]

    # Back Substitution
    M = [0.0] * n
    for i in reversed(range(n)):
        s = sum(A[i][j] * M[j] for j in range(i+1, n))
        M[i] = (B[i] - s) / A[i][i]
    return M

def interpolate_x(x, y, M0_x=0, Mn_x=0):
    x_dum = x
    y_dum = y
    spline_num = len(x) - 1
    # Compute distances between successive points
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    # Arc-length parameter
    t = np.zeros(len(x))
    t[1:] = np.cumsum(distances) 

    y = x
    x = t

    n = spline_num  # Number of spline segments

    A = [[0.0 for _ in range(n+1)] for _ in range(n+1)]
    B = [0.0 for _ in range(n+1)]

    h = [x[i+1] - x[i] for i in range(n)]

    # Boundary conditions
    A[0][0] = 1.0
    B[0] = M0_x
    A[n][n] = 1.0
    B[n] = Mn_x

    # Interior equations
    for i in range(1, n):
        h_prev = h[i-1]
        h_next = h[i]

        A[i][i-1] = h_prev
        A[i][i] = 2 * (h_prev + h_next)
        A[i][i+1] = h_next

        B[i] = 6 * ((y[i+1] - y[i]) / h_next - (y[i] - y[i-1]) / h_prev)

    # Solve the system A·M = B

    import copy
    M = gaussian_elimination(copy.deepcopy(A), B)

    # Compute spline coefficients a, b, c, d
    a_x = []
    b_x = []
    c_x = []
    d_x = []

    for i in range(n):  
        a_x.append(y[i])
        b_x.append((y[i+1] - y[i]) / h[i] - h[i]*(2*M[i] + M[(i+1)]) / 6)
        c_x.append(M[i] / 2)
        d_x.append((M[(i+1)] - M[i]) / (6 * h[i]))

    coeffs_x = np.zeros((spline_num, 4))
    for i in range((spline_num)):
        coeffs_x[i][0] = d_x[i]
        coeffs_x[i][1] = c_x[i]
        coeffs_x[i][2] = b_x[i]
        coeffs_x[i][3] = a_x[i] 
    splines_x = []
    for i in range(spline_num):
        splines_x.append(polynomial(coeffs_x[i]))
    
    x = x_dum
    y = y_dum

    splines_dx = []
    splines_dx_dx = []
    for i in range(spline_num):
        # Compute the first derivative of the polynomial
        d_x = splines_x[i].derivative()
        splines_dx.append(d_x)
        # Compute the second derivative of the polynomial
        d_x_dx = d_x.derivative()
        splines_dx_dx.append(d_x_dx)

    first_dervatives = []
    second_dervatives = []
    for i in range(spline_num + 1):
        if i == spline_num:
            first_dervatives.append(splines_dx[spline_num - 1].__call__(t[i] - t[i-1]))
            second_dervatives.append(splines_dx_dx[spline_num - 1].__call__(t[i] - t[i-1]))
        else:
            first_dervatives.append(splines_dx[i].__call__(0))
            second_dervatives.append(splines_dx_dx[i].__call__(0))

    return splines_x, x, t, first_dervatives, second_dervatives


def interpolate_y(x, y, M0_y = 0, Mn_y = 0):
    x_dum = x
    y_dum = y
    spline_num = len(x) - 1
    # Compute distances between successive points
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    # Arc-length parameter
    t = np.zeros(len(x))
    t[1:] = np.cumsum(distances) 

    x = t

    n = spline_num  # Number of spline segments

    A = [[0.0 for _ in range(n+1)] for _ in range(n+1)]
    B = [0.0 for _ in range(n+1)]

    h = [x[i+1] - x[i] for i in range(n)]

    # Boundary conditions
    A[0][0] = 1.0
    B[0] = M0_y
    A[n][n] = 1.0
    B[n] = Mn_y

    # Interior equations
    for i in range(1, n):
        h_prev = h[i-1]
        h_next = h[i]

        A[i][i-1] = h_prev
        A[i][i] = 2 * (h_prev + h_next)
        A[i][i+1] = h_next

        B[i] = 6 * ((y[i+1] - y[i]) / h_next - (y[i] - y[i-1]) / h_prev)

    # Solve the system A·M = B

    import copy
    M = gaussian_elimination(copy.deepcopy(A), B)

    # Compute spline coefficients a, b, c, d
    a_x = []
    b_x = []
    c_x = []
    d_x = []

    for i in range(n):
        a_x.append(y[i])
        b_x.append((y[i+1] - y[i]) / h[i] - h[i]*(2*M[i] + M[(i+1)]) / 6)
        c_x.append(M[i] / 2)
        d_x.append((M[(i+1)] - M[i]) / (6 * h[i]))

    coeffs_x = np.zeros((spline_num, 4))
    for i in range((spline_num)):
        coeffs_x[i][0] = d_x[i]
        coeffs_x[i][1] = c_x[i]
        coeffs_x[i][2] = b_x[i]
        coeffs_x[i][3] = a_x[i] 
    splines_x = []
    for i in range(spline_num):
        splines_x.append(polynomial(coeffs_x[i]))
    
    x = x_dum
    y = y_dum
    splines_dy = []
    splines_dy_dy = []
    for i in range(spline_num):
        # Compute the first derivative of the polynomial
        d_x = splines_x[i].derivative()
        splines_dy.append(d_x)
        # Compute the second derivative of the polynomial
        d_x_dx = d_x.derivative()
        splines_dy_dy.append(d_x_dx)

    first_dervatives = []
    second_dervatives = []  
    for i in range(spline_num + 1):
        if i == spline_num:
            first_dervatives.append(splines_dy[spline_num - 1].__call__(t[i] - t[i-1]))
            second_dervatives.append(splines_dy_dy[spline_num - 1].__call__(t[i] - t[i-1]))
        else:   
            first_dervatives.append(splines_dy[i].__call__(0))
            second_dervatives.append(splines_dy_dy[i].__call__(0))

    return splines_x, y, t, first_dervatives, second_dervatives
