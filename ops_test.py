import numpy as np

from node import Node
from node import MatrixMultiply
from executor import Executor
from utils import var, gradients

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
    
def test_add_mul_mix_2():
    x1 = var("x1")
    x2 = var("x2")
    x3 = var("x3")
    x4 = var("x4")
    y = x1 + x2 * x3 * x4
    
    grad_x1, grad_x2, grad_x3, grad_x4 = gradients(y, [x1, x2, x3, x4])
    executor = Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
    
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x4_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val, x3 : x3_val, x4 : x4_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))
    assert np.array_equal(grad_x2_val, x3_val * x4_val)
    assert np.array_equal(grad_x3_val, x2_val * x4_val)
    assert np.array_equal(grad_x4_val, x2_val * x3_val)
    
def test_grad_of_grad():
    x2 = var("x2")
    x3 = var("x3")
    y = x2 * x2 + x2 * x3
    
    grad_x2, grad_x3 = gradients(y, [x2, x3])
    grad_x2_x2, grad_x2_x3 = gradients(grad_x2, [x2, x3])

    executor = Executor([y, grad_x2, grad_x3, grad_x2_x2, grad_x2_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val, grad_x2_x2_val, grad_x2_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = x2_val * x2_val + x2_val * x3_val
    expected_grad_x2_val = 2 * x2_val + x3_val 
    expected_grad_x3_val = x2_val
    expected_grad_x2_x2_val = 2 * np.ones_like(x2_val)
    expected_grad_x2_x3_val = 1 * np.ones_like(x2_val)

    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)
    assert np.array_equal(grad_x2_x2_val, expected_grad_x2_x2_val)
    assert np.array_equal(grad_x2_x3_val, expected_grad_x2_x3_val)
    

def test_mul_by_const():
    x2 = var("x2")
    y = 5 * x2

    grad_x2, = gradients(y, [x2])

    executor = Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x2_val * 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val) * 5)

def test_matmul_two_vars():
    x2 = var("x2")
    x3 = var("x3")
    mm = MatrixMultiply()
    y = mm(x2, x3)

    grad_x2, grad_x3 = gradients(y, [x2, x3])
    
    executor = Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]]) 
    x3_val = np.array([[7, 8, 9], [10, 11, 12]]) 

    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)