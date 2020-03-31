from typing import List
import numpy as np
import tvm_op
import topi

from utils import broadcast_rule

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
    
    def compute(self, node, vals, output, compiled_func):
        pass

    def gradient(self, node, grad):
        pass
    
    def infer_shape(self, node, shapes):
        pass
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
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
    
    def compute(self, node, vals, output, compiled_func):
        compiled_func(vals[0], vals[1], output)
    
    def gradient(self, node, grad):
        return [grad, grad] # Contribution of each value to the gradient
    
    def infer_shape(self, node, shape):
        return broadcast_rule(shape[0], shape[1])
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return tvm_op.element_wise_addition(shapes[0], "element_wise_addn")

class AddConst(BaseOp):
    
    def __call__(self, node1, val):
        node = BaseOp.__call__(self)
        node.desc = "(%s + %s)" % (node1.desc, str(val))
        node.const_attribute = val
        node.inputs = [node1]
        
        return node
    
    def compute(self, node, vals, output, compiled_func):
        compiled_func(vals[0], output)
    
    def gradient(self, node, grad):
        return [grad]
    
    def infer_shape(self, node, shape):
        const = (1,)
        return broadcast_rule(shape[0], const)
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return tvm_op.element_wise_addition_by_const(shapes[0], node.const_attribute, "elemwise_add_const")

class Multiply(BaseOp):
    
    def __call__(self, node1, node2):
        node = BaseOp.__call__(self)
        node.desc = "(%s * %s)" % (node1.desc, node2.desc)
        node.inputs = [node1, node2]
        return node
    
    def compute(self, node, vals, output, compiled_func):
        compiled_func(vals[0], vals[1], output)
    
    def gradient(self, node, grad):
        return [node.inputs[1] * grad, node.inputs[0] * grad]
    
    def infer_shape(self, node, shape):
        return broadcast_rule(shape[0], shape[1])
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return tvm_op.element_wise_mul(shapes[0], "element_wise_multiplication")
 
        
class MultiplyByConst(BaseOp):
    def __call__(self, node1, val):
        node = BaseOp.__call__(self)
        node.const_attribute = val
        node.inputs = [node1]
        node.desc = "(%s + %s)" % (node1.desc, str(val))
        return node
    
    def compute(self, node, vals, output, compiled_func):
        compiled_func(vals[0], output) 
    
    def gradient(self, node, grad):
        return [node.const_attribute * grad]
    
    def infer_shape(self, node, shape):
        c_shape = (1,)
        return broadcast_rule(shape[0], c_shape)
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return tvm_op.element_wise_mul_by_const(shapes[0], node.const_attribute, "element_wise_multiplication_byconst")
    
class Placeholder(BaseOp):
    def __call__(self):
        node = BaseOp.__call__(self)
        return node
    
    def compute(self, node, vals, output, compiled_func):
        pass
    
    def gradient(self, node, grad):
        return None

    def infer_shape(self, node, shape):
        pass
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return None


class ZerosLike(BaseOp):
    def __call__(self, node1):
        node = BaseOp.__call__(self)
        node.inputs = [node1]
        node.desc = "zeros like shape of (%s)" % node1.desc
        return node
    
    def compute(self, node, vals, output, compiled_func):
        output.copyfrom(np.zeros(vals[0].shape, dtype = vals[0].dtype))
    
    def gradient(self, node, grad):
        temp = ZerosLike()
        return [temp(node.inputs[0])]

    def infer_shape(self, node, shape):
        return shape[0]
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return None

class OnesLikeOp(BaseOp):
    def __call__(self, node1):
        node = BaseOp.__call__(self)
        node.inputs = [node1]
        node.desc = "ones like shape of (%s)" % node1.desc
        return node
    
    def compute(self, node, vals, output, compiled_func):
        output.copyfrom(np.ones(vals[0].shape, dtype = vals[0].dtype))
    
    def gradient(self, node, grad):
        temp = ZerosLike()
        return [temp(node.inputs[0])]

    def infer_shape(self, node, shape):
        return (1,)
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return None
    
class ReduceSumAxis(BaseOp):
    def __call__(self, node1):
        node = BaseOp.__call__(self)
        node.inputs = [node1]
        node.desc = "ReduceSumAxis (%s)" % (node1.desc)
        return node

    def compute(self, node, vals, output, compiled_func):
        compiled_func(vals[0], output)
    
    def gradient(self, node, grad):
        temp = BroadcastTo()
        return [temp(grad, node.inputs[0])]

    def infer_shape(self, node, shape):
        if (len(shape[0]) == 1):
            return (1,)
        return shape[0][1:]
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return tvm_op.reduce_sum_axis_zero(shapes[0], "reduce_sum_over_axis")
    
class BroadcastTo(BaseOp):
    def __call__(self, node1, node2):
        node = BaseOp.__call__(self)
        node.inputs = [node1, node2]
        node.desc = "BroadcaseOp (%s, %s.shape)" % (node1.desc, node2.desc)
        return node
    
    def compute(self, node, vals, output, compiled_func):
        compiled_func(vals[0], output)
    
    def gradient(self, node, grad):
        temp1 = ReduceSumAxis()
        temp2 = ZerosLike()
        grad_A = temp1(grad)
        grad_B = temp2(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, shape):
        return broadcast_rule(shape[0], shape[1])
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return tvm_op.broadcast_to(shapes[0], shapes[1], "broadcast_op")
    
class MatrixMultiply(BaseOp):
    def __call__(self, node1, node2, t_1 = False, t_2 = False):
        node = BaseOp.__call__(self)
        node.inputs = [node1, node2]
        node.desc = "(%s, %s, %s, %s)" % (node1.desc, node2.desc, str(t_1), str(t_2))
        node.transpose_1 = t_1
        node.transpose_2 = t_2
        return node
    
    def compute(self, node, vals, output, compiled_func):
        compiled_func(vals[0], vals[1], output)
    
    def gradient(self, node, grad):
        matmul_op = MatrixMultiply()
        if ((node.transpose_1 is False) and (node.transpose_2 is False)):
            lhs_grad = matmul_op(grad, node.inputs[1], False, True)
            rhs_grad = matmul_op(node.inputs[0], grad, True, False)
        elif ((node.transpose_1 is True) and (node.transpose_2 is False)):
            lhs_grad = matmul_op(node.inputs[1], grad, False, True)
            rhs_grad = matmul_op(node.inputs[0], grad, True, False)
        elif ((node.transpose_1 is False) and (node.transpose_2 is True)):
            lhs_grad = matmul_op(grad, node.inputs[1], False, True)
            rhs_grad = matmul_op(grad, node.inputs[0], True, False)
        elif ((node.transpose_1 is True) and (node.transpose_2 is True)):
            lhs_grad = matmul_op(node.inputs[1], grad, False, True)
            rhs_grad = matmul_op(grad, node.inputs[0], True, False)
        return [lhs_grad, rhs_grad]
    
    def infer_shape(self, node, shape):
        assert(len(shape[0]) == 2 and len(shape[1]) == 2)
        l = shape[0]
        r = shape[1]
        if node.transpose_1:
            l = (shape[0][1], shape[0][0])
        if node.transpose_2:
            r = (shape[1][1], shape[1][0])
        assert(l[1] == r[0])
        return (l[0], r[1])
    
    def compiled_func(self, node, shapes, tgt, tgt_host):
        return tvm_op.matrix_multiply(shapes[0], node.transpose_1, shapes[1], node.transpose_2, "matrix_mult")
    
    