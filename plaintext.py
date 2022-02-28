import numpy as np
from data import *
# problem: A @ x = b
# derivative: 2 * A.T @ (A @ x - b) = 2*A.T@A@X - 2*A.T@b
# Hyperparams
num_data = 20
x_dim = 5
x_group = 1

a_dim = (num_data, x_dim)
x_dim = (x_dim, x_group)
b_dim = (num_data, x_group)

iter_step = 50
l_rate = .0005

# Generate some data
# A = np.random.randn(a_dim[0], a_dim[1]) * 4
# x = np.random.randn(x_dim[0], x_dim[1]) * 4
# b = A @ x

# Now x is unknown. We will do some calculations:
x_iter = np.random.randn(x_dim[0], x_dim[1]) * 4
for idx in range(iter_step):
    grad = 2 * A.T @ (A @ x_iter - b)
    x_iter = x_iter - l_rate * grad

print(f"x_iter = {x_iter.T}")
print(f"real_x = {x.T}")
print(f"Difference = {np.linalg.norm(x - x_iter)}")