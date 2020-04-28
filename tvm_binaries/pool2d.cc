
#include "ops.h"


Op Model::get_or_create_pool2d(Tensor _input, OpBase::OpType _type,
                               int _kernelH, int _kernelW,
                               int _strideH, int _strideW,
                               int _padH, int _padW, bool _relu)
{

  Pool2DKey key(_input, _type, _kernelH, _kernelW, _strideH, _strideW,
                _padH, _padW, _relu);
  Pool2D* poolOp;
  if (pool2d.find(key) != pool2d.end()) {
    poolOp = pool2d[key];
  } else {
    poolOp = new Pool2D(this, _input, _type, _kernelH, _kernelW,
                        _strideH, _strideW, _padH, _padW, _relu);
    measure_pool2d_cost(poolOp);
    pool2d[key] = poolOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = poolOp;
  return ret;
}

Tensor Graph::pool2d_max(Tensor _input,
                         int _kernelH, int _kernelW,
                         int _strideH, int _strideW,
                         int _padH, int _padW, bool _relu)
{
  Op op = model->get_or_create_pool2d(
              _input, OpBase::OP_POOL2D_MAX, _kernelH, _kernelW,
              _strideH, _strideW, _padH, _padW, _relu);
  inEdges[op];
  outEdges[op];
  Edge in(_input.idx, _input.op), out(_input.idx, op);
  inEdges[op].insert(in);
  outEdges[_input.op].insert(out);
  Tensor t = op.ptr->outputs[0];
  t.op = op;
  return t;
}

Pool2D::Pool2D(Model* _model, Tensor _input, OpType _type,
               int _kernelH, int _kernelW,
               int _strideH, int _strideW,
               int _padH, int _padW, bool _relu)
: OpBase(_input, _model, _type),
  kernelH(_kernelH), kernelW(_kernelW),
  strideH(_strideH), strideW(_strideW), 
  padH(_padH), padW(_padW), relu(_relu)
{
  assert(type == OP_POOL2D_MAX || type == OP_POOL2D_AVG);
  assert(_input.numDim == 4);
  int inputC = _input.dim[1];
  int inputH = _input.dim[2];
  int inputW = _input.dim[3];
  int outputH = 1 + (inputH + 2 * padH - kernelH) / strideH;
  int outputW = 1 + (inputW + 2 * padW - kernelW) / strideW;
#ifdef VERBOSE
  printf("k(%d %d) pad(%d %d) s(%d %d)\n",
         kernelH, kernelW, padH, padW, strideH, strideW);
#endif
  numOutputs = 1;
  outputs[0].numDim = 4;
  outputs[0].dim[0] = BATCH_SIZE;
  outputs[0].dim[1] = inputC;
  outputs[0].dim[2] = outputH;
  outputs[0].dim[3] = outputW;
  outputs[0].idx = 0;
}

Pool2D::~Pool2D(void)
{
}

bool Pool2D::get_parameter(OpParameter para, int* value)
{
  switch (para) {
    case PM_OP_TYPE:
      *value = (int) type;
      return true;
    case PM_NUM_INPUTS:
      *value = numInputs;
      return true;
    case PM_NUM_OUTPUTS:
      *value = numOutputs;
      return true;
    case PM_KERNEL_H:
      *value = kernelH;
      return true;
    case PM_KERNEL_W:
      *value = kernelW;
      return true;
    case PM_STRIDE_H:
      *value = strideH;
      return true;
    case PM_STRIDE_W:
      *value = strideW;
      return true;
    case PM_PAD_H:
      *value = padH;
      return true;
    case PM_PAD_W:
      *value = padW;
      return true;
    case PM_RELU:
      *value = (int) relu;
      return true;
    default:
      return false;
  }
}

void Pool2D::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  int outputSize = 1, inputSize = 1;
  for (int i = 0; i < outputs[0].numDim; i++)
    outputSize *= outputs[0].dim[i];
  for (int i = 0; i < inputs[0].numDim; i++)
    inputSize *= inputs[0].dim[i];
  exe_time += runtime;
  flops += outputSize * kernelH * kernelW;
  mem_acc += inputSize;
  num_kernels += 1;
}

Pool2DKey::Pool2DKey(Tensor _input, OpBase::OpType _type,
                     int _kernelH, int _kernelW, int _strideH, int _strideW,
                     int _padH, int _padW, bool _relu)
{
 keys[0] = _input.dim[0];
 keys[1] = _input.dim[1];
 keys[2] = _input.dim[2];
 keys[3] = _input.dim[3];
 keys[4] = _kernelH;
 keys[5] = _kernelW;
 keys[6] = _strideH;
 keys[7] = _strideW;
 keys[8] = _padH;
 keys[9] = _padW;
 keys[10] = (int)(_relu);
 keys[11] = (int)(_type);
}

