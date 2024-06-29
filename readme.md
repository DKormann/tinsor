
## you like [tinygrad](https://github.com/tinygrad/tinygrad)? you know [einsum notation](https://rockt.github.io/2018/04/30/einsum)?


```python
from tinsor import Shape

T, U = Shape(T=3, U=4)

w = T.U.rand() # random tensor

w.sum() # float

w + w # tensor
```