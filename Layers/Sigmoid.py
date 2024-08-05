import numpy as np 

class Sigmoid():
    def __init__(self):
        pass
    def forward(self,x):
        return 1/(1+np.e**-x)
        





if __name__ == '__main__':
    smd = Sigmoid()
    lst = [0.6,0.1,0.5,0.1,0.4]
    array = np.array(lst)
    output = smd.forward(array)
    print(output)
