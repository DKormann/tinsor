#%%
from eintensor import shape


T, U = shape(T=3, U=4)

T, U

x = T.U.ones()
w = T.U.rand()

square = T.T.ones()

res = x + w

res.shape
