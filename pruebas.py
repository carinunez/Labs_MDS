import numpy as np

a = np.array([[1,2,3],[1,2,3]])
b = a + np.array([[1],[3]])
print(b==4)
print((b==4).astype(int))