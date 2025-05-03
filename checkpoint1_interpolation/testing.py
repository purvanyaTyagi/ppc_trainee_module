import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from final_func import polynomial, gaussian_elimination, interpolate_x, interpolate_y

df= pd.read_csv('loop_track_waypoints.csv')

df = df.iloc[10:]

x = np.array(df['X'])
y = np.array(df['Y'])

splines_x, x, t, first_derivatives_x, second_derivatives_x = interpolate_x(x, y, 0, 0)
splines_y, y, t, first_derivatives_y, second_derivatives_y = interpolate_y(x, y, 0, 0) 

n = len(x) - 1

import matplotlib.pyplot as plt

# Plot original points
plt.scatter(t, x, color='red', label='Data Points')

# Plot each spline segment
for i in range(n):
    xi = t[i]
    xi1 = t[i+1]  # next point, x[49+1] = x[0] due to cyclic nature


    xs = [xi + (xi1 - xi) * t / 100.0 for t in range(101)]
    ys = [splines_x[i].__call__(x_val - xi) for x_val in xs]
    plt.plot(xs, ys, color='blue')
    plt.xlim(0, 1)

plt.xlim(0, 1)
plt.title("Cyclic Cubic Spline Interpolation")
plt.xlabel("x")
plt.ylabel("t")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()


plt.scatter(t, y, color='red', label='Data Points')

# Plot each spline segment
for i in range(n):
    xi = t[i]
    xi1 = t[i+1]  # next point, x[49+1] = x[0] due to cyclic nature


    xs = [xi + (xi1 - xi) * t / 100.0 for t in range(101)]
    ys = [splines_y[i].__call__(x_val - xi) for x_val in xs]
    plt.plot(xs, ys, color='blue')
    plt.xlim(0, 1)

plt.xlim(0, 1)
plt.title("Cyclic Cubic Spline Interpolation")
plt.xlabel("y")
plt.ylabel("t")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

plt.scatter(x, y, color='red', label='Data Points')


for i in range(n):
    ti = t[i]
    ti1 = t[i+1]  # next point, x[49+1] = x[0] due to cyclic nature


    ts = [ti + (ti1 - ti) * t / 100.0 for t in range(101)]

    ys = [splines_y[i].__call__(x_val - ti) for x_val in ts]
    xs = [splines_x[i].__call__(x_val - ti) for x_val in ts]
    plt.plot(xs, ys, color='blue')

plt.xlim(0, 1)
plt.title("Cyclic Cubic Spline Interpolation")
plt.xlabel("x")
plt.ylabel("t")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

print(len(second_derivatives_x))
print(len(second_derivatives_y))
print(len(x))
print(second_derivatives_x)
print(second_derivatives_y)