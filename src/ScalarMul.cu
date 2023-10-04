#include <ScalarMul.cuh>

/*
 * Calculates scalar multiplication for block
 */


__global__
void ScalarMulBlock(int numElements, float *vector1, float *vector2, float *result) {
  extern __shared__ float shared_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  shared_data[tid] = 0;

  if (index < numElements) {
    shared_data[tid] = vector1[index] * vector2[index];

  }
  __syncthreads();

  for (size_t i = (blockDim.x / 2); i > 0; i >>= 1) {
    if (tid < i) {
      shared_data[tid] += shared_data[tid + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, shared_data[0]);
  }
}
