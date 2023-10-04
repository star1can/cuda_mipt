#include <KernelMul.cuh>

#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <fstream>

auto CreateHostArray(size_t array_size) {
  auto *array = (float *) malloc(array_size * sizeof(float));
  assert(array != nullptr);

  for (size_t i = 0; i < array_size; ++i) {
    if (i % 2 == 0) {
      array[i] = 1.0f;
    } else {
      array[i] = 0.f;
    }
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

void Run(size_t N, size_t block_size, std::ofstream &results) {

  auto host_arrays = CreateHostArrays(N, 3);
  auto device_arrays = CreateDeviceArrays(N, host_arrays);

  size_t blocks_count = (N + block_size - 1) / block_size;

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  KernelMul<<<blocks_count, block_size>>>(N, device_arrays[0], device_arrays[1], device_arrays[2]);

  cudaEventRecord(stop);
  cudaMemcpy(host_arrays[2], device_arrays[2], N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  float total_time = 0;

  cudaEventElapsedTime(&total_time, start, stop);
  results << N << " " << block_size << " " << total_time << "\n";

  FreeDeviceMemory(device_arrays);
  FreeHostMemory(host_arrays);
}

int main() {
  std::ofstream results;
  results.open("task2");

  for (size_t N = 1 << 10; N <= (1 << 28); N <<= 3) {
    for (size_t block_size = 32; block_size < 2048; block_size *= 2) {
      Run(N, block_size, results);
    }
  }

  results.close();

}
