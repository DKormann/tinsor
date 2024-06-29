#%%

from tinsor import Shape, Tensor, Dim



#%%

S, T, U, V = Shape(S=5, T=3, U=4, V=6)

x = Tensor.ones(S, T)
y = Tensor.rand(T, U)

p = x @ y

print(p)

p = x * y

print(p)

e = p.expand(V)

e

# %%

from dataclasses import dataclass


@dataclass(eq=False, frozen=True)
class T:
  x:int

  def __new__(cls, x:int):
    # return super().__new__(cls
    return x


T(33)
# %%
