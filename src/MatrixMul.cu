#include <MatrixMul.cuh>

__global__
void MatrixMul(int heightA, int widthA, int widthB, float *matrixA, float *matrixB, float *matrixResult) {

  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t column = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ float sub_matrices[];

  size_t dim = blockDim.x;
  size_t tid_x = threadIdx.x;
  size_t tid_y = threadIdx.y;

  float* sh_a = sub_matrices;
  float* sh_b = sub_matrices + dim * dim;

  float sum = 0.f;


  for (size_t i = 0; i  < widthA; i += dim) {
    if ((i + tid_x < widthA) && (row < heightA)) {
      sh_a[tid_y * dim + tid_x] = matrixA[row * widthA + i + tid_x];
    } else {
      sh_a[tid_y * dim + tid_x] = 0.f;
    }

    if ((i + tid_y < widthA) && (column < widthB)) {
      sh_b[tid_y * dim + tid_x] = matrixB[i * widthB + tid_y * widthB + column];
    } else {
      sh_b[tid_y * dim + tid_x] = 0.f;
    }

    __syncthreads();
    for (size_t j = 0; j < dim; ++j) {
      sum += sh_a[tid_y * dim + j] * sh_b[j * dim + tid_x];
    }
    __syncthreads();
  }


  if ((row < heightA) && (column < widthB)) {
    matrixResult[row * widthB + column] = sum;
  }
}

