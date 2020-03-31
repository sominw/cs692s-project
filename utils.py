from typing import List
from operator import add
from functools import reduce
import numpy as np

from node import Placeholder, OnesLikeOp

def sum_nodes(nodes):
    return reduce(add, nodes)

def softmax_fn(y):
    x = y - np.max(y, axis=1, keepdims=True)
    exp_x = np.exp(x)
    softmax = exp_x / np.sum(exp_x, axis = 1, keepdims=True)
    return softmax

def var(desc):
    pn = Placeholder()
    pn.desc = desc
    return pn

def topological_sort(node, visited, order):
    if node in visited:
        return 
    visited.add(node)
    for nde in node.inputs:
        topological_sort(nde, visited, order)
    order.append(node)
    
def topological_sort_lookup(nodes):
    visited = set()
    order = list()
    for node in nodes:
        topological_sort(node, visited, order)
    return order

def gradients(node, node_list):
    node_to_grad = dict()
    temp = OnesLikeOp()
    reverse_topo = reversed(topological_sort_lookup([node]))
    node_to_grad[node] = [temp(node)]
    node_to_grad_ = dict()
    
    for n in reverse_topo:
        grad = sum_nodes(node_to_grad[n])
        node_to_grad_[n] = grad
        input_grad = n.op.gradient(n, grad)
        for i in range(len(n.inputs)):
            if n.inputs[i] not in node_to_grad:
                node_to_grad[n.inputs[i]] = list()
            node_to_grad[n.inputs[i]].append(input_grad[i])
    
    grad_list = [node_to_grad_[n] for n in node_list]
    return grad_list

def broadcast_rule(shape_a, shape_b):
    
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)