dict = {}
for each in ["1","2","1","3"]:
    if each in dict.keys():
        dict[each]+=1
    else:
        dict[each]=1
sorted(dict_data.items(),key=lambda x:x[1])
print(list(dict.items())[:2])
exit()
x = "123141"
print(list(x))
exit()

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
