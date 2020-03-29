from typing import List
import numpy as np

from op import Add, AddConst, Multiply, MultiplyByConst, Placeholder, ZerosLike, OnesLikeOp, MatrixMultiply

class Node:
   
    def __init__(self):
        self.op = None
        self.const_attribute = None
        self.desc = ""
        self.inputs = list()
        
    def __add__(self, input):
        if isinstance(input, Node):
            add = Add()
            return add(self, input)
        else:
            temp = AddConst()
            return temp(self, input) 
    
    def __mul__(self, input):
        if isinstance(input, Node):
            mul = Multiply()
            return mul(self, input)
        else:
            temp = MultiplyByConst()
            return temp(self, input)
    
    def __str__(self):
        return self.desc
    
    __repr__ = __str__
    __radd__ = __add__ 
    __rmul__ = __mul__

