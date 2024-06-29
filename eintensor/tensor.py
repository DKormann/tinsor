#%%
from dataclasses import dataclass
from typing import Any
import tinygrad
import tinygrad.tensor

# %%
@dataclass(eq=False, frozen=True)
class Dim:
  name: str
  size: int

  def __repr__(self): return f'Dim("{self.name}", {self.size})'
  def __getattribute__(self, name: str) -> Any:
    try: return super().__getattribute__(name)
    except AttributeError: return Shape(self).__getattr__(name)
  

dimdict = {}
def dim(name:str, n:int):
  assert name not in ['name', 'size'], f'{name} is a reserved name'
  assert name not in dimdict, f'{name} already defined'
  dimdict[name] = Dim(name, n)
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
  def keys(self): return [k for k,v in self.shape]
  
  def ones(self): return EinTensor(self, tinygrad.Tensor.ones(self.size))
  def rand(self): return EinTensor(self, tinygrad.Tensor.rand(self.size))
  def zeros(self): return EinTensor(self, tinygrad.Tensor.zeros(self.size))
  def eye(self): return EinTensor(self, tinygrad.Tensor.eye(self.size[0]))

  def __getattr__(self, key): return Shape(*self.dims,dimdict[key])

  
def shape(**kwargs): return Shape(tuple (Dim(k,v) for k,v in kwargs.items()))

class EinTensor:
  shape: Shape
  tensor: tinygrad.Tensor

  def __init__(self, shape: Shape, tensor: tinygrad.Tensor):
    self.shape = shape
    self.tensor = tensor

  def numpy(self): return self.tensor.numpy()

  def __repr__(self): return f'<Tensor {self.shape} {self.tensor.dtype} device={self.tensor.device}>'

  def __getattr__(self, key):
    for k,v in self.shape.shape:
      if k == key: return v
    raise AttributeError(f'{key} not found in {self.shape}')

  def sum(self, axis:Shape):
    for k in axis.keys: assert k in self.shape.keys, f'{k} not in {self.shape}'
    axes = [self.shape.keys.index(k) for k in axis.keys]
    newshape = tuple((k,v) for k,v in self.shape.shape if k not in axis.keys)
    return EinTensor(Shape(newshape), self.tensor.sum(axes))

  def __add__(self, other: 'EinTensor'):
    return EinTensor(self.shape, self.tensor + other.tensor)


T, U = Shape(U=3, T=4)

tensor = T.U.ones()

print(tensor)


