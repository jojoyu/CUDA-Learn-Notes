import numpy as np

matrix = np.arange(0, 12).reshape(3, 4)

print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print(f"sum axis=0 {np.sum(matrix, axis=0)}")
# [12 15 18 21]
print(f"sum axis=1 {np.sum(matrix, axis=1)}")
# [ 6 22 38]
print(f"max axis=0 {np.max(matrix, axis=0)}")
print(f"max axis=1 {np.max(matrix, axis=1)}")
