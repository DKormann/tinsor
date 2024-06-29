#%%
from dataclasses import dataclass
import tinygrad
import tinygrad.tensor

# %%

@dataclass(eq=False, frozen=True)
class Shape():
  shape: tuple[tuple[str, int]] = ()
  
  def __repr__(self):
    return f'Shape[{", ".join([f"{k}:{v}" for k,v in self.shape])}]'
  
  def __iter__(self):
    return tuple(Shape({k:v}) for k,v in self.shape)
  
  @property
  def size(self): return tuple([v for k,v in self.shape])

  @property
  def keys(self): return [k for k,v in self.shape]
  
  def ones(self): return EinTensor(self, tinygrad.Tensor.ones(self.size))
  def rand(self): return EinTensor(self, tinygrad.Tensor.rand(self.size))
  def zeros(self): return EinTensor(self, tinygrad.Tensor.zeros(self.size))
  def eye(self): return EinTensor(self, tinygrad.Tensor.eye(self.size[0]))

  def __getattr__(self, key):
    val = globals()[key].shape
    assert len(val) == 1, f'{key} is not a scalar'
    while key in self.keys: key = '_' + key
    return Shape(self.shape + ((key, val[0][1]),))

  
def shape(**kwargs):
  return Shape(tuple(kwargs.items()))

@dataclass(eq=False, frozen=True)
class EinTensor:
  shape: Shape
  tensor: tinygrad.Tensor

  def numpy(self): return self.tensor.numpy()

  def __repr__(self): return f'Tensor[{", ".join([f"{k}:{v}" for k,v in self.shape.shape])}]'

  def __getattr__(self, key):
    for k,v in self.shape.shape:
      if k == key: return v
    raise AttributeError(f'{key} not found in {self.shape}')

  def sum(self, axis:Shape):
    for k in axis.keys: assert k in self.shape.keys, f'{k} not in {self.shape}'
    axes = [self.shape.keys.index(k) for k in axis.keys]
    newshape = tuple((k,v) for k,v in self.shape.shape if k not in axis.keys)
    return EinTensor(Shape(newshape), self.tensor.sum(axes))

T, U = shape(T=4, U=3)

x = T.U.rand()
z = T.U.ones()

z.sum(T).numpy()

