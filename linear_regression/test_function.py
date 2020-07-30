s = {"b":2,"x":3,"c":4}
print([s])
exit()

import numpy as np
w = np.random.normal(1,0.1)
b = np.random.normal(1,0.1)
print("w:",w)
print("b",b)
x = np.array([1.1,3.3,5.5])
y = x * 20 +6
print("x",x)
print(w*x+b)
print((w*x+b)-y)
print(((w*x+b)-y)*x)
