#%%
from tinsor import Shape

S, T, U, V = Shape(S=5, T=3, U=4, V=6)

x = S.T.U.ones()
y = U.V.S.rand()

x * y
