import pickle
import numpy as np
a = np.array([1, 2, 3])

with open("./testdump.npy", "rb") as f:
    b = pickle.load(f)

print(b)
print(a)