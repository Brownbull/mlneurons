# based on
# https://junjiecai.github.io/en/posts/2017/Jan/31/sympy_intro_4/
# %matplotlib widget
from sympy import *
from sympy.plotting import plot3d_parametric_surface

x, y = symbols('x y')
f1 = 3*cos(x) + cos(x)*cos(y)
f2 = 3*sin(x) + sin(x)*cos(y)
f3 = sin(y)

x_range = (x, 0, 2*pi)
y_range = (y, 0, 2*pi)

plot3d_parametric_surface(f1, f2, f3, x_range, y_range)
