#include <CommonKernels.cuh>

__global__
void Reduce(int numElements, float *array, float *result) {
  extern __shared__ float shared_data[];

  size_t tid = threadIdx.x;
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  shared_data[tid] = 0;

  if(index < numElements) {
    shared_data[tid] = array[index];
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