#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float *A, float *B, float *result) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t column = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < height) && (column < width)) {
    result[pitch * row + column] = A[pitch * row + column] + B[pitch* row + column];
  }
}
