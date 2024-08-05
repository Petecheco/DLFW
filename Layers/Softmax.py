import numpy as np 



class Softmax():
    def __init__(self,dim=None):
        assert dim != None, "You have to specify a specific dim for the softmax operation(at least for now :)"
        self.dim = dim

    def forward(self,x):
        num_of_dims = x.ndim
        assert self.dim<=num_of_dims, "The dim must be less or equal to the actual dimension of the data"
        exponential = np.e**(x)
        sum_value = np.sum(exponential,self.dim)
        return exponential / sum_value






if __name__ == '__main__':
    smx = Softmax(dim=0)
    data = np.random.randn(3,3,3)
    print(data)
    outputs = smx.forward(data)
    print(sum(outputs[:,2,2]))
