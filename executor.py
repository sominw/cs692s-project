import numpy as np
import tvm
import topi

from node import Node, Placeholder
from utils import topological_sort_lookup

class Executor:
    
    def __init__(self, node_list, ctx=None):
        
        self.eval_list = node_list
        self.ctx = ctx
        if self.ctx == tvm.cpu(0):
            self.tgt = "llvm"
            self.tgt_host = "llvm"
        else:
            print ("Error executing on non-CPU contexts")
        self.topo_order = topological_sort_lookup(self.eval_list)
        self.node_to_arr = None
        self.node_to_shape = None
        self.node_to_compiled_func = None
        self.feed_shapes = None
        
    def infer_shape(self, feed_shapes):
        self.node_to_shape = dict()
        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape[node] = feed_shapes[node]
                continue
            shapes = [self.node_to_shape[n] for n in node.inputs]
            self.node_to_shape[node] = node.op.infer_shape(node, shapes)
    
    def memory_plan(self, feed_shapes):
        self.node_to_arr = dict()
        for node in self.topo_order:
            if node in feed_shapes:
                continue
            self.node_to_arr[node] = tvm.runtime.ndarray.empty(self.node_to_shape[node], dtype="float32", ctx=self.ctx)
            
    
    def compile_funcs(self, feed_shapes):
        self.node_to_compiled_func = dict()
        for node in self.topo_order:
            if node in feed_shapes:
                continue
            input_shapes = [self.node_to_shape[n] for n in node.inputs]
            self.node_to_compiled_func[node] = node.op.compile_func(node, input_shapes, self.tgt, self.tgt_host)
            
        
    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        
        node_to_val = dict()
        
        def feed_shapes_eq(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            ui = set(sa.items()) ^ set(sb.items())
            return len(ui) == 0
        
        for n, v in feed_dict.items():
            node_to_val[n] = v
        
        feed_shapes = dict()
        for node in node_to_val:
            feed_shapes[node] = node_to_val[node].shape
            
        if (not feed_shapes_eq(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            self.memory_plan(feed_shapes)
            self.compile_funcs(feed_shapes)
            
        for node in self.topo_order:
            if node in node_to_val:
                continue
            input_vals = [node_to_val[n] for n in node.inputs]
            node_val = self.node_to_arr[node]
            node.op.compute(node, input_vals, node_val, self.node_to_compiled_func[node])
            node_to_val[node] = node_val
        
        if (convert_to_numpy_ret_vals):
            return [node_to_val[n].asnumpy() for n in self.eval_list]
        return [node_to_val[n] for n in self.eval_list]