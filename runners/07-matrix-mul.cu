#include <MatrixMul.cuh>

#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

auto CreateHostArray(size_t array_size) {
  auto *array = (float *) malloc(array_size * sizeof(float));
  assert(array != nullptr);
  for (size_t i = 0; i < array_size; ++i) {
    if (i % 2 == 0) {
      array[i] = 4;
    } else {
      array[i] = 2;
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

void Run(size_t height_a,
         size_t width_a,
         size_t width_b,
         size_t thread_count_1,
         size_t thread_count_2,
         std::ofstream &results) {

  size_t height_b = width_a;
  float *h_a = CreateHostArray(height_a * width_a);
  float *h_b = CreateHostArray(height_b * width_b);
  float *h_result = CreateHostArray(height_a * width_b);

  float *d_a = CreateDeviceArray(h_a, height_a * width_a);
  float *d_b = CreateDeviceArray(h_b, height_b * width_b);
  float *d_result = CreateDeviceArray(h_result, height_a * width_b);

  dim3 block_size(thread_count_1, thread_count_2);
  dim3 grid_size((width_b + thread_count_1 - 1) / thread_count_1, (height_a + thread_count_2 - 1) / thread_count_2);

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  MatrixMul<<<grid_size, block_size, 2 * thread_count_1 * thread_count_2 * sizeof(float)>>>(height_a,
                                                                                            width_a,
                                                                                            width_b,
                                                                                            d_a,
                                                                                            d_b,
                                                                                            d_result);

  cudaEventRecord(stop);
  cudaMemcpy(h_result, d_result, height_a * width_b * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  float *seq_res = CreateHostArray(height_a * width_b);

  for (size_t i = 0; i < height_a; ++i) {
    for (size_t j = 0; j < width_b; ++j) {

      float sum = 0.f;
      for (size_t k = 0; k < width_a; ++k) {
        sum += h_a[i * width_a + k] * h_b[k * width_b + j];
      }

      seq_res[i * width_b + j] = sum;
    }
  }

  float max_error = 0.f;
  for (size_t i = 0; i < height_a; ++i) {
    for (size_t j = 0; j < width_b; ++j) {
      max_error = fmax(max_error, fabs(seq_res[i * width_b + j] - h_result[i * width_b + j]));
    }
  }

  float total_time = 0;

  cudaEventElapsedTime(&total_time, start, stop);

  results << height_a << " " << width_a << " " << height_b << " " << width_b << " " << thread_count_1 << " "
          << thread_count_2 << " " << max_error << " " << total_time << std::endl;

  FreeHostArray(seq_res);
  FreeHostArray(h_a);
  FreeHostArray(h_b);
  FreeHostArray(h_result);

  FreeDeviceArray(d_a);
  FreeDeviceArray(d_b);
  FreeDeviceArray(d_result);
}

int main() {
  std::ofstream results;
  results.open("task7");

  for (size_t height_a = 1 << 2; height_a <= (1 << 10); height_a <<= 2) {
    for (size_t width_a = 1 << 2; width_a <= (1 << 10); width_a <<= 2) {
      for (size_t width_b = 1 << 2; width_b <= (1 << 10); width_b <<= 2) {
        for (size_t thread_count_1 = 4; thread_count_1 <= 32; thread_count_1 *= 2) {
          Run(height_a, width_a, width_b, thread_count_1, thread_count_1, results);
        }
      }
    }
  }

  // Run(1024, 512, 13, 16, 16, results);
  results.close();

}

