from typing import List
from operator import add
from functools import reduce

def sum_nodes(nodes):
    return reduce(add, nodes)

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