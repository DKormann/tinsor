#%%
from dataclasses import dataclass

from typing import Any
import tinygrad
import tinygrad.tensor
from tinygrad.helpers import dedup

# %%
@dataclass(eq=False, frozen=True)
class Dim:
  name: str
  size: int

  def __repr__(self): return f'Dim("{self.name}", {self.size})'
  def __getattribute__(self, name: str) -> 'Shape':
    try: return super().__getattribute__(name)
    except AttributeError: return Shape(self).__getattr__(name)

dimdict:dict[str, Dim] = {}
def dim(name:str, n:int):
  if name in dimdict: assert dimdict[name].size==n, f'{name} already defined as {dimdict[name]}'
  else: dimdict[name] = Dim(name, n)
  return dimdict[name]

class Shape():
  dims: tuple[Dim]

  def __init__(self,*dims,**kwargs):
    self.dims = dims + tuple(dim(k, v) for k,v in kwargs.items())
  
  def __repr__(self): return f'Shape({", ".join([f"{d.name}={d.size}" for d in self.dims])})'
  def __iter__(self): return (d for d in self.dims)
  
  @property
  def size(self): return tuple([d.size for d in self.dims])

  @property
  def keys(self): return [d.name for d in self.dims]
  
  def ones(self): return EinTensor(self, tinygrad.Tensor.ones(self.size))
  def rand(self): return EinTensor(self, tinygrad.Tensor.rand(self.size))
  def zeros(self): return EinTensor(self, tinygrad.Tensor.zeros(self.size))
  def eye(self): return EinTensor(self, tinygrad.Tensor.eye(self.size[0]))

  def __getattr__(self, key): return Shape(*self.dims,dimdict[key])

class EinTensor:
  shape: Shape
  tensor: tinygrad.Tensor

  def __init__(self, shape: Shape, tensor: tinygrad.Tensor):
    self.shape = shape
    self.tensor = tensor

  def numpy(self): return self.tensor.numpy()

  def __repr__(self): return f'<Tensor {self.shape} {self.tensor.dtype} device={self.tensor.device}>'

  def sum(self) -> float:
    return self.tensor.sum().numpy().item()

  def __add__(self, other: 'EinTensor'):
    return EinTensor(self.shape, self.tensor + other.tensor)


  def binary_fn(self, other: 'EinTensor', fn: Any) -> 'EinTensor':
    newshape = Shape(*dedup(self.shape.dims + other.shape.dims))

    stensor = self.tensor
    otensor = other.tensor

    for k in newshape.keys:
      if k not in self.shape.keys: stensor = stensor.unsqueeze(-1)
      if k not in other.shape.keys: otensor = otensor.unsqueeze(-1)

    last = len(newshape.keys)
    perm = [other.shape.keys.index(k) if k in other.shape.keys else (last:=last-1) for k in newshape.keys]
    otensor = otensor.permute(*perm)

    last = len(newshape.keys)
    perm = [self.shape.keys.index(k) if k in self.shape.keys else (last:=last-1) for k in newshape.keys]
    stensor = stensor.permute(*perm)

    return EinTensor(newshape, fn(stensor, otensor))
  
  def __add__(self, other: 'EinTensor'): return self.binary_fn(other, tinygrad.Tensor.add)
  def __sub__(self, other: 'EinTensor'): return self.binary_fn(other, tinygrad.Tensor.sub)
  def __mul__(self, other: 'EinTensor'): return self.binary_fn(other, tinygrad.Tensor.mul)
  def __truediv__(self, other: 'EinTensor'): return self.binary_fn(other, tinygrad.Tensor.div)
  
    
S, T, U, V = Shape(S=5, T=3, U=4, V=6)

x = S.T.U.ones()
y = U.V.S.rand()


x * y


