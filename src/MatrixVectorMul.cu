#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, float *matrix, float *vector, float *result) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < height) {
    float sum =0.f;
    for (size_t i = 0; i < width; ++i) {
      sum += matrix[width * index + i] * vector[i];
    }
    result[index] = sum;
  }
}

