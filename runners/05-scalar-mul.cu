#include <ScalarMulRunner.cuh>

#include <iostream>
#include <cassert>
#include <vector>

auto CreateHostArray(size_t array_size) {
  auto *array = (float *) malloc(array_size * sizeof(float));
  assert(array != nullptr);

  for (int i = 0; i < array_size; ++i) {
    array[i] = 1;
  }

  return array;
}

auto CreateHostArrays(size_t array_size, size_t array_count) {

  std::vector<float *> host_arrays(array_count, nullptr);
  for (auto &host_array: host_arrays) {
    host_array = CreateHostArray(array_size);
  }

  for (int i = 0; i < array_size; ++i) {
    host_arrays[array_count - 1][i] = 0.f;
  }

  return host_arrays;
}

auto CreateDeviceArray(float *host_array, size_t array_size) {
  float *array = nullptr;
  cudaMalloc(&(array), array_size * sizeof(float));
  assert(array != nullptr);

  cudaMemcpy(array, host_array, array_size * sizeof(float), cudaMemcpyHostToDevice);
  return array;
}

auto CreateDeviceArrays(size_t array_size, const std::vector<float *> &host_arrays) {
  std::vector<float *> device_arrays(host_arrays.size(), nullptr);

  for (size_t i = 0; i < host_arrays.size(); ++i) {
    device_arrays[i] = CreateDeviceArray(host_arrays[i], array_size);
  }

  return device_arrays;
}

void FreeHostArray(float *array) {
  free(array);
}

void FreeDeviceArray(float *array) {
  cudaFree(array);
}

void FreeHostMemory(std::vector<float *> &arrays) {
  for (auto &array: arrays) {
    FreeHostArray(array);
  }
}

void FreeDeviceMemory(std::vector<float *> &arrays) {
  for (auto &array: arrays) {
    FreeDeviceArray(array);
  }
}

void RunTwoReductions(size_t array_size, size_t block_size) {
  float *vector_1 = CreateHostArray(array_size);
  float *vector_2 = CreateHostArray(array_size);

  float scalar_product = ScalarMulTwoReductions(array_size, vector_1, vector_2, block_size);

  FreeHostArray(vector_1);
  FreeHostArray(vector_2);
}

void RunOneReduction(size_t array_size, size_t block_size) {
  float *vector_1 = CreateHostArray(array_size);
  float *vector_2 = CreateHostArray(array_size);

  float scalar_product = ScalarMulSumPlusReduction(array_size, vector_1, vector_2, block_size);

  FreeHostArray(vector_1);
  FreeHostArray(vector_2);
}

int main() {
  for (size_t N = 1 << 10; N <= (1 << 28); N <<= 3) {
    for (size_t block_size = 32; block_size < 2048; block_size *= 2) {
      RunTwoReductions(N, block_size);
    }
  }

  for (size_t N = 1 << 10; N <= (1 << 28); N <<= 3) {
    for (size_t block_size = 32; block_size < 2048; block_size *= 2) {
      RunOneReduction(N, block_size);
    }
  }
}

