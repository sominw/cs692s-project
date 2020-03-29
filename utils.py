from typing import List
from operator import add
from functools import reduce

from node import Placeholder, OnesLikeOp

def sum_nodes(nodes):
    return reduce(add, nodes)

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
            node_to_grad[n.inputs[i]] = node_to_grad.get(n.inputs[i], [])
            node_to_grad[n.inputs[i]].append(input_grad[i])
    
    grad_list = [node_to_grad_[n] for n in node_list]
    return grad_list