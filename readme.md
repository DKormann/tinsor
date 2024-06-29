
## you like [tinygrad](https://github.com/tinygrad/tinygrad)? you know [einsum notation](https://rockt.github.io/2018/04/30/einsum)?

This tensor frontend will take care of all unsqueese / permutes you need


```python
# define your data dimensions
S, T, U, V = Shape(S=5, T=3, U=4, V=6)

# create completely different tensors
x = S.T.U.ones()
y = U.V.S.rand()

# let tinsor match the dimensions new shape: (S=5, T=3, U=4, V=6)
x * y
```