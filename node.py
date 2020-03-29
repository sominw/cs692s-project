from typing import List
import numpy as np

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


class BaseOp:
    
    def compute(self, node, vals):
        pass

    def gradient(self, node, grad):
        pass
    
    def __call__(self):
        node = Node()
        node.op = self
        return node
    
class Add(BaseOp):
    
    def __call__(self, node1, node2):
        node = BaseOp.__call__(self)
        node.desc = str(node1.desc) + str(" + ") + str(node2.desc)
        node.inputs = [node1, node2]
        return node
    
    def compute(self, node: Node, vals):
        return vals[0] + vals[1]
    
    def gradient(self, node: Node, grad):
        return [grad, grad] # Contribution of each value to the gradient
    

class AddConst(BaseOp):
    
    def __call__(self, node1, val):
        node = BaseOp.__call__(self)
        node.desc = "(%s + %s)" % (node1.desc, str(val))
        node.const_attribute = val
        node.inputs = [node1]
        
        return node
    
    def compute(self, node, val):
        return val[0] + node.const_attribute
    
    def gradient(self, node, grad):
        return [grad]
    

class Multiply(BaseOp):
    
    def __call__(self, node1, node2):
        node = BaseOp.__call__(self)
        node.desc = "(%s * %s)" % (node1.desc, node2.desc)
        node.inputs = [node1, node2]
        return node
    
    def compute(self, node, val):
        return val[0] * val[1]
    
    def gradient(self, node, grad):
        mul = Multiply()
#        res = [Multiply(node.inputs[1], grad), Multiply(node.inputs[0], grad)]
        res = [mul(node.inputs[1], grad), mul(node.inputs[0], grad)]
        return res
        
class MultiplyByConst(BaseOp):
    def __call__(self, node1, val):
        node = BaseOp.__call__(self)
        node.const_attribute = val
        node.inputs = [node1]
        node.desc = "(%s + %s)" % (node1.desc, str(val))
        return node
    
    def compute(self, node, val):
        return val[0] * node.const_attribute
    
    def gradient(self, node, grad):
        mulc = MultiplyByConst()
        res = [mulc(grad, node.const_attribute)]
        return res
    
class Placeholder(BaseOp):
    def __call__(self):
        node = BaseOp.__call__(self)
        return node
    def compute(self, node, val):
        pass
    def gradient(self, node, grad):
        return None

class ZerosLike(BaseOp):
    def __call__(self, node1):
        node = BaseOp.__call__(self)
        node.inputs = [node1]
        node.desc = "zeros like shape of (%s)" % node1.desc
        return node
    
    def compute(self, node, val):
        return np.zeros(val[0].shape)
    
    def gradient(self, node, grad):
        temp = ZerosLike()
        return [temp(node.inputs[0])]
    
class OnesLikeOp(BaseOp):
    def __call__(self, node1):
        node = BaseOp.__call__(self)
        node.inputs = [node1]
        node.desc = "ones like shape of (%s)" % node1.desc
        return node
    
    def compute(self, node, vals):
        return np.ones(vals[0].shape)
    
    def gradient(self, node, grad):
        temp = ZerosLike()
        return [temp(node.inputs[0])]
        
    
class MatrixMultiply(BaseOp):
    def __call__(self, node1, node2, t_1 = False, t_2 = False):
        node = BaseOp.__call__(self)
        node.inputs = [node1, node2]
        node.desc = "(%s, %s, %s, %s)" % (node1.desc, node2.desc, str(t_1), str(t_2))
        node.transpose_1 = t_1
        node.transpose_2 = t_2
        return node
    
    def compute(self, node, val):
        if (node.transpose_1):
            val[0] = val[0].T
        if (node.transpose_2):
            val[1] = val[1].T
        return np.dot(val[0], val[1])
    
    def gradient(self, node, grad):
        temp = MatrixMultiply()
        res = [temp(grad, node.inputs[1], False, True), temp(node.inputs[0], grad, True, False)]
        return res