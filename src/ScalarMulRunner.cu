#include <ScalarMulRunner.cuh>

#include <ScalarMul.cuh>
#include <KernelMul.cuh>
#include <CommonKernels.cuh>
#include <cassert>

float ScalarMulTwoReductions(int numElements, float *vector1, float *vector2, int blockSize) {

  float *d_vector_1 = nullptr;
  float *d_vector_2 = nullptr;
  float *d_res = nullptr;

  size_t blocks_count = (numElements + blockSize - 1) / blockSize;

  cudaMalloc(&d_vector_1, numElements * sizeof(float));
  cudaMalloc(&d_vector_2, numElements * sizeof(float));
  cudaMalloc(&d_res, sizeof(float));
  cudaMemset(d_res, 0.f, sizeof(float));

  assert(d_vector_1 != nullptr);
  assert(d_res != nullptr);
  assert(d_vector_2 != nullptr);

  cudaMemcpy(d_vector_1, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_2, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);

  ScalarMulBlock<<<blocks_count, blockSize, sizeof(float) * blockSize>>>(numElements,
                                                                         d_vector_1,
                                                                         d_vector_2,
                                                                         d_res);

  float scalar_product = 0.f;
  cudaMemcpy(&scalar_product, d_res, sizeof(float), cudaMemcpyDeviceToHost);

  float total_time = 0;

  float seq_scalar_product = 0.f;

  for (size_t i = 0; i < numElements; ++i) {
    seq_scalar_product += vector1[i] * vector2[i];
  }

  cudaFree(d_vector_1);
  cudaFree(d_vector_2);
  cudaFree(d_res);

  return scalar_product;
}

float ScalarMulSumPlusReduction(int numElements, float *vector1, float *vector2, int blockSize) {

  float *d_vector_1 = nullptr;
  float *d_vector_2 = nullptr;
  float *d_product = nullptr;
  float *d_res = nullptr;

  size_t blocks_count = (numElements + blockSize - 1) / blockSize;

  cudaMalloc(&d_vector_1, numElements * sizeof(float));
  cudaMalloc(&d_vector_2, numElements * sizeof(float));
  cudaMalloc(&d_product, numElements * sizeof(float));
  cudaMalloc(&d_res, sizeof(float));
  cudaMemset(d_res, 0.f, sizeof(float));

  assert(d_vector_1 != nullptr);
  assert(d_res != nullptr);
  assert(d_vector_2 != nullptr);
  assert(d_product != nullptr);

  cudaMemcpy(d_vector_1, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_2, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);

  KernelMul<<<blocks_count, blockSize>>>(numElements, d_vector_1, d_vector_2, d_product);

  cudaDeviceSynchronize();

  Reduce<<<blocks_count, blockSize, sizeof(float) * blockSize>>>(numElements,
                                                                 d_product,
                                                                 d_res);

  float scalar_product = 0.f;

  cudaMemcpy(&scalar_product, d_res, sizeof(float), cudaMemcpyDeviceToHost);

  float total_time = 0;

  float seq_scalar_product = 0.f;

  for (size_t i = 0; i < numElements; ++i) {
    seq_scalar_product += vector1[i] * vector2[i];
  }

  cudaFree(d_vector_1);
  cudaFree(d_vector_2);
  cudaFree(d_res);
  cudaFree(d_product);

  return scalar_product;
}
