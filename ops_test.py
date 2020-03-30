import numpy as np


from node import Node
from executor import Executor
from utils import gradients, var

def test_var():
    x1 = var("x1")
    y = x1

    grad_x1, = gradients(y, [x1])

    executor = Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val= executor.run(feed_dict = {x1 : x1_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))