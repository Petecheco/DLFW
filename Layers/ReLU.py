import numpy as np 
"""
Some basic implementation of ReLU activation function.

$y=\cases{
x,x\geq0 \\
0,x\lt0
}$
"""

class ReLU():
    def __init__(self):
        pass

    def forward(self,x):
        return x * (x>0)



if __name__ == '__main__':
    activation = ReLU()
    lst = [0.5,16.8,-1.6,0.8]
    array = np.array(lst)

    print(f"The input before the activation {array}")
    output = activation.forward(array)
    print(f"The final output is {output}")
