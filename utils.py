from typing import List
from operator import add
from functools import reduce

from node import Placeholder, OnesLikeOp

def sum_nodes(nodes):
    return reduce(add, nodes)

def var(desc):
    pn = Placeholder()
    pn_node = pn()
    pn_node.desc = desc
    return pn_node

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

"""
def gradients(output_node, node_list):
    node_to_output_grads_list = {}
    temp = OnesLikeOp()
    node_to_output_grads_list[output_node] = [temp(output_node)]
    node_to_output_grad = {}
    reverse_topo_order = reversed(topological_sort_lookup([output_node]))

    for node in reverse_topo_order:
        grad = sum_nodes(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad
        input_grads = node.op.gradient(node, grad)
        for i in range(len(node.inputs)):
            node_to_output_grads_list[node.inputs[i]] = node_to_output_grads_list.get(node.inputs[i], [])
            node_to_output_grads_list[node.inputs[i]].append(input_grads[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list
"""

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