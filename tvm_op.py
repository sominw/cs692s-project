import numpy as np
import tvm
import topi

from __future__ import print_function, absolute_import

def reduce_sum_axis_zero(shape, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def sgd_update(shape, learning_rate, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    X = tvm.te.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.te.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.te.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.te.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f

def broadcast_to(shape, to_shape, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def element_wise_addition(shape, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + B(*i))
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def element_wise_addition_by_const(shape, const_k, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.tir.const(const_k, A.dtype)
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + B)
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def element_wise_mul(shape, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) * B(*i))
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def element_wise_mul_by_const(shape, const_k, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.tir.const(const_k, A.dtype)
    C = tvm.te.compute(A.shape, lambda *i: A(*i) * B)
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def relu(shape, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.tir.const(0, A.dtype)
    C = tvm.te.compute(A.shape, lambda *i: tvm.tir.max(A(*i), B))
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def relu_grad(shape, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.tir.const(0, A.dtype)
    D = tvm.te.compute(A.shape, lambda *i: tvm.tir.expr.Select((A(*i) > C), B(*i), C))
    s = tvm.te.create_schedule(D.op)
    f = tvm.build(s, [A, B, D], tgt, target_host=tgt_host, name=func_name)
    return f

def matrix_multiply(shapeA, transposeA, shapeB, transposeB, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    A = tvm.te.placeholder((shapeA[0], shapeA[1]), dtype=dtype, name="A")
    B = tvm.te.placeholder((shapeB[0], shapeB[1]), dtype=dtype, name="B")

    if transposeA == False and transposeB == False:
        k = tvm.te.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.te.compute((shapeA[0], shapeB[1]), lambda i, j: tvm.tir.sum(A[i, k] * B[k, j], axis=k))
    elif transposeA == True and transposeB == False:
        k = tvm.te.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.te.compute((shapeA[1], shapeB[1]), lambda i, j: tvm.tir.sum(A[k, i] * B[k, j], axis=k))
    elif transposeA == False and transposeB == True:
        k = tvm.te.reduce_axis((0, shapeA[1]), name='k')
        C = tvm.te.compute((shapeA[0], shapeB[0]), lambda i, j: tvm.tir.sum(A[i, k] * B[j, k], axis=k))
    else:
        k = tvm.te.reduce_axis((0, shapeA[0]), name='k')
        C = tvm.te.compute((shapeA[1], shapeB[0]), lambda i, j: tvm.tir.sum(A[k, i] * B[j, k], axis=k))

    s = tvm.te.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=64)
    xk, yk = s[C].split(k, factor=8)
    s[C].reorder(xo, yo, xk, xi, yi, yk)
    s[C].parallel(xo)
    s[C].unroll(yk)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def conv2d(shapeX, shapeF, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF
    Input = tvm.te.placeholder(shapeX, dtype=dtype, name="A")
    Filter = tvm.te.placeholder(shapeF, dtype=dtype, name="B")
    di = tvm.te.reduce_axis((0, R), name='di')
    dj = tvm.te.reduce_axis((0, S), name='dj')
    dc = tvm.te.reduce_axis((0, C), name='dc')
    Output = tvm.te.compute((N, M, H - R + 1, W - S + 1), lambda n, m, i, j: tvm.tir.sum(Input[n, dc, i + di, j + dj] * Filter[m, dc, di, dj], axis=[di, dj, dc]), name='Output')
    s = tvm.te.create_schedule(Output.op)
    f = tvm.build(s, [Input, Filter, Output], tgt, target_host=tgt_host, name=func_name)
    return f

def matrix_softmax(shape, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    k = tvm.te.reduce_axis((0, shape[1]), name="k")
    max_A = tvm.te.compute((shape[0],), lambda i: tvm.tir.max(A[i, k], axis=k), name="max_A")
    exp = tvm.te.compute(shape, lambda i, j: tvm.tir.exp(A[i, j] - max_A[i]), name="exp")
    k1 = tvm.te.reduce_axis((0, shape[1]), name="k1")
    sum_exp = tvm.te.compute((shape[0],), lambda i: tvm.tir.sum(exp[i, k1], axis=k1), name="sum_exp")
    B = tvm.te.compute(shape, lambda i, j: exp[i, j] / sum_exp[i], name="B")
    s = tvm.te.create_schedule(B.op)
    f = tvm.build(s, [A, B], tgt, target_host=tgt_host, name=func_name)
    return f

def matrix_cross_entropy(shape, func_name, dtype="float32", tgt="llvm", tgt_host="llvm"):
    
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    k = tvm.te.reduce_axis((0, shape[1]), name="k")
    max_A = tvm.te.compute((shape[0],), lambda i: tvm.tir.max(A[i, k], axis=k), name="max_A")
    exp = tvm.te.compute(shape, lambda i, j: tvm.exp(A[i, j] - max_A[i]), name="exp")
    k1 = tvm.te.reduce_axis((0, shape[1]), name="k1")
    sum_exp = tvm.te.compute((shape[0],), lambda i: tvm.tir.sum(exp[i, k1], axis=k1), name="sum_exp")
    softmax = tvm.te.compute(shape, lambda i, j: exp[i, j] / sum_exp[i], name="softmax")

    log = tvm.te.compute(shape, lambda i, j: tvm.log(softmax[i, j]), name = "log")
    k2 = tvm.te.reduce_axis((0, shape[1]), name="k2")
    sum_softmax = tvm.te.compute((shape[0],), lambda i: tvm.tir.sum(B[i, k2] * log[i, k2], axis = k2), name="sum_softmax")
    k3 = tvm.te.reduce_axis((0, shape[0]), name="k3")
    softmax_cross_entropy = tvm.te.compute((1,), lambda i: tvm.tir.sum(-1 * sum_softmax[k3] / shape[0], axis = k3))

    s = tvm.te.create_schedule(softmax_cross_entropy.op)
    f = tvm.build(s, [A, B, softmax_cross_entropy], tgt, target_host=tgt_host, name=func_name)
    return f
