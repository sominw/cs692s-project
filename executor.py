import numpy as np

from node import Node
from node import Placeholder
from utils import topological_sort_lookup

class Executor:
    
    def __init__(self, node_list):
        
        self.eval_list = node_list
        
    def run(self, feed_dict):
        
        node_to_val = dict(feed_dict)
        topo_order = topological_sort_lookup(self.eval_list)
        
        for node in topo_order:
            if (isinstance(node.op, Placeholder)):
                continue
            input_v = [node_to_val[i] for i in node.inputs]
            res = node.op.compute(node, input_v)
            if (isinstance(res, np.ndarray) == False):
                res = np.array(res)
            node_to_val[node] = res
        
        results = [node_to_val[node] for node in self.eval_list]
        return results