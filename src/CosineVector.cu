#include <CosineVector.cuh>
#include <ScalarMul.cuh>

#include <cassert>
#include <vector>

float CosineVector(int numElements, float *vector1, float *vector2, int blockSize) {
  float *d_vector_1 = nullptr;
  float *d_vector_2 = nullptr;

  std::vector<float *> d_products(3, nullptr);

  for (auto &d_product: d_products) {
    cudaMalloc(&d_product, 3 * sizeof(float));
    assert(d_product != nullptr);
    cudaMemset(d_product, 0.f, sizeof(float));
  }

  size_t blocks_count = (numElements + blockSize - 1) / blockSize;

  cudaMalloc(&d_vector_1, numElements * sizeof(float));
  cudaMalloc(&d_vector_2, numElements * sizeof(float));

  assert(d_vector_1 != nullptr);
  assert(d_vector_2 != nullptr);

  cudaMemcpy(d_vector_1, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_2, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);

  ScalarMulBlock<<<blocks_count, blockSize, sizeof(float) * blockSize>>>(numElements,
                                                                         d_vector_1,
                                                                         d_vector_1,
                                                                         d_products[0]);

  ScalarMulBlock<<<blocks_count, blockSize, sizeof(float) * blockSize>>>(numElements,
                                                                         d_vector_2,
                                                                         d_vector_2,
                                                                         d_products[1]);

  ScalarMulBlock<<<blocks_count, blockSize, sizeof(float) * blockSize>>>(numElements,
                                                                         d_vector_1,
                                                                         d_vector_2,
                                                                         d_products[2]);

  cudaDeviceSynchronize();

  std::vector<float> dot_products(3, 0.f);

  for (size_t i = 0; i < d_products.size(); ++i) {
    cudaMemcpy(&(dot_products[i]), d_products[i], sizeof(float), cudaMemcpyDeviceToHost);
  }

  float norm_product = dot_products[0] * dot_products[1];
  float cosine = (norm_product != 0) ? dot_products[2] / norm_product : 0;

  cudaFree(d_vector_1);
  cudaFree(d_vector_2);
  for (auto &d_product: d_products) {
    cudaFree(d_product);
  }
  return cosine;
}
