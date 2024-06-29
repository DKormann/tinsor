#%%
from tinsor import Shape

T, U = Shape(T=3, U=4)

w = T.U.rand() # random tensor

w.sum() # float

w + w # tensor



