import tenseal as ts
import numpy as np

from context import *
from data import *
context = gencontext()
print("Generation complete!")

# Hyperparams
num_data = 20
x_dim = 5
x_group = 1

a_dim = (num_data, x_dim)
x_dim = (x_dim, x_group)
b_dim = (num_data, x_group)

iter_step = 100
l_rate = .002

# Generate some data
# A = np.random.randn(a_dim[0], a_dim[1]) * 4
# x = np.random.randn(x_dim[0], x_dim[1]) * 4
x_iter = np.random.randn(x_dim[0], x_dim[1]) * 4
# b = A @ x

# Encrypt A, b and an unknown x

enc_A = encrypt(context, A)
enc_AT = encrypt(context, 2 * l_rate * A.T)
enc_b = encrypt(context, b)

enc_x_iter = encrypt(context, x_iter)

# Now x is unknown. We will do some calculations:

for idx in range(iter_step):
    enc_grad = enc_AT @ (enc_A @ enc_x_iter - enc_b)# 2 * A.T @ (A @ x_iter - b)
    enc_x_iter = enc_x_iter - enc_grad
    if idx % 4 == 0:
        enc_x_iter = bootstrap(context, enc_x_iter)
    if idx % 4 == 0:
        print("-----------------------------")
        print(f"Iter {idx} complete!")
        print(f"target_x = {x.T}")
        print(f"current_x = {decrypt(enc_x_iter).T}")

print(f"x_iter = {x_iter.T}")
x_iter_after = decrypt(enc_x_iter)
print(f"x_iter_after = {x_iter_after.T}")

print(f"target_x = {x.T}")
print(f"Difference = {np.linalg.norm(x - x_iter_after)}")

print(x_iter_after.T@A.T-b.T)

for idx in range(iter_step):
    grad = 2 * A.T @ (A @ x_iter - b)
    x_iter = x_iter - l_rate * grad

print(f"x_iter = {x_iter.T}")
print(f"real_x = {x.T}")
print(f"Difference = {np.linalg.norm(x - x_iter)}")