#include <MatrixVectorMul.cuh>

#include <cassert>
#include <fstream>
#include <vector>

auto CreateHostArray(size_t array_size) {
  auto *array = (float *) malloc(array_size * sizeof(float));
  assert(array != nullptr);

  for (size_t i = 0; i < array_size; ++i) {
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

void Run(size_t height, size_t width, size_t block_size, std::ofstream &results) {
  auto *host_vector = CreateHostArray(width);
  auto *host_vector_res = CreateHostArray(height);

  auto *device_vector = CreateDeviceArray(host_vector, width);
  auto *device_vector_res = CreateDeviceArray(host_vector_res, height);

  auto *host_matrix = CreateHostArray(height * width);
  auto *device_matrix = CreateDeviceArray(host_matrix, height * width);

  size_t grid_size = (height + block_size - 1) / block_size;

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  MatrixVectorMul<<<grid_size, block_size>>>(height,
                                             width,
                                             device_matrix,
                                             device_vector,
                                             device_vector_res);

  cudaEventRecord(stop);
  cudaMemcpy(host_vector_res, device_vector_res, height * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  float total_time = 0;

  cudaEventElapsedTime(&total_time, start, stop);

  auto *seq_res = (float *) malloc(sizeof(float) * height);

  for (size_t i = 0; i < height; ++i) {
    float sum = 0.f;
    for (size_t j = 0; j < width; ++j) {
      sum += host_matrix[i * width + j] * host_vector[j];
    }
    seq_res[i] = sum;
  }

  float max_error = 0.f;
  for (size_t i = 0; i < height; ++i) {
    max_error = fmax(max_error, fabs(seq_res[i] - host_vector_res[i]));
  }

  results << height << " " << width << " " << block_size << " " << max_error << " " << total_time << std::endl;

  FreeDeviceArray(device_vector_res);
  FreeDeviceArray(device_vector);
  FreeDeviceArray(device_matrix);

  FreeHostArray(host_vector_res);
  FreeHostArray(host_vector);
  FreeHostArray(seq_res);
}

int main() {
  std::ofstream results;
  results.open("task4");

  for (size_t height = 1 << 2; height <= (1 << 10); height <<= 2) {
    for (size_t width = 1 << 2; width <= (1 << 10); width <<= 2) {
      for (size_t block_size = 32; block_size <= 1024; block_size *= 2) {
        Run(height, width, block_size, results);
      }
    }
  }

  results.close();
}

