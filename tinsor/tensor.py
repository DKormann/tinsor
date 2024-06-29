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
  def __getattribute__(self, name: str) -> 'Shape':
    try: return super().__getattribute__(name)
    except AttributeError: return Shape(self).__getattr__(name)
  

dimdict:dict[str, Dim] = {}
def dim(name:str, n:int):
  assert name not in ['name', 'size'], f'{name} is a reserved name'
  assert name not in dimdict or dimdict[name].size==n, f'{name} already defined as {dimdict[name]}'
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

  def sum(self) -> float:
    return self.tensor.sum().numpy().item()

  def __add__(self, other: 'EinTensor'):
    return EinTensor(self.shape, self.tensor + other.tensor)
