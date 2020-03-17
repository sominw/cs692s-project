from node import Node

class BaseOp:
    
    def compute(self, node : Node, vals):
        pass

    def gradient(self, node: Node, grad):
        pass
    
    def __call__(self):
        node = Node()
        node.op = self
        return node
    
class Add(BaseOp):
    
    def __call__(self, node1: Node, node2: Node):
        node = BaseOp.__call__(self)
        node.desc = str(node1.desc) + str(" + ") + str(node2.desc)
        node.inputs = [node1, node2]
        return node
    
    def compute(self, node: Node, vals):
        return vals[0] + vals[1]
    
    def grad(self, node: Node, grad):
        return [grad, grad] # Contribution of each value to the gradient
    

 