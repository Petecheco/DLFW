import numpy as np 
"""
A simple implementation of fully connected layer using numpy
$ y = x A^T + b  $
Assume the input shape is (N,C_in)
The weight matrix shape is (C_in, C_out)
"""

class Linear():
    def __init__(self,in_channels, out_channels, bias=False, init="uniform"):
        assert init in ["uniform", "kaiming"], "unsupported initialization type!"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.if_bias = bias
        self.init = init
        self.alpha = -np.sqrt(1/self.in_channels)
        self.weight = np.random.uniform(self.alpha,-self.alpha,(self.in_channels,self.out_channels))
        self.bias = np.random.uniform(self.alpha,-self.alpha,(self.out_channels)) if self.if_bias else 0

    def forward(self, x):
        input_dim = x.shape[1]
        assert input_dim==self.in_channels, f"The input dimension should match the weight dimension as {self.in_channels} instead of {input_dim}'"
        outputs = x@self.weight + self.bias
        return outputs


if __name__=='__main__':
    fc = Linear(64,128)
    x = np.random.randn(10,128)
    result = fc.forward(x)
    print(result)
