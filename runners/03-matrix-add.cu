#include <KernelMatrixAdd.cuh>

#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>

auto CreateHostMatrix(size_t height, size_t width) {
  auto *matrix = (float *) malloc(height * width * sizeof(float));
  assert(matrix != nullptr);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      matrix[i * width + j] = i + j;
    }
  }

  return matrix;
}

auto CreateHostMatrices(size_t height, size_t width, size_t matrix_count) {
  std::vector<float *> matrices(matrix_count, nullptr);

  for (auto &matrix: matrices) {
    matrix = CreateHostMatrix(height, width);
    assert(matrix != nullptr);
  }

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      matrices[matrix_count - 1][i * width + j] = 0;
    }
  }

  return matrices;
}

auto CreatePitchedDeviceMatrix(float *host_matrix, size_t height, size_t width, size_t *pitch) {
  float *matrix = nullptr;
  cudaMallocPitch(&(matrix), pitch, width * sizeof(float), height);
  assert(matrix != nullptr);
  cudaMemcpy2D(matrix,
               *pitch,
               host_matrix,
               width * sizeof(float),
               width * sizeof(float),
               height,
               cudaMemcpyHostToDevice);
  return matrix;
}

auto CreateDeviceMatrices(const std::vector<float *> host_matrices,
                          size_t height,
                          size_t width,
                          size_t *pitch) {
  std::vector<float *> matrices(host_matrices.size(), nullptr);

  for (size_t i = 0; i < matrices.size(); ++i) {
    matrices[i] = CreatePitchedDeviceMatrix(host_matrices[i], height, width, pitch);
  }

  return matrices;
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

void Run(size_t height, size_t width, size_t threads_count_1, size_t threads_count_2, std::ofstream &out) {
  size_t pitch = 0;

  auto host_matrices = CreateHostMatrices(height, width, 3);
  auto device_matrices = CreateDeviceMatrices(host_matrices, height, width, &pitch);

  dim3 block_size(threads_count_1, threads_count_2);
  dim3 grid_size((width + threads_count_1 - 1) / block_size.x, (height + threads_count_2 - 1) / block_size.y);

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  KernelMatrixAdd<<<grid_size, block_size>>>(height,
                                             width,
                                             pitch / sizeof(float),
                                             device_matrices[0],
                                             device_matrices[1],
                                             device_matrices[2]);

  cudaEventRecord(stop);
  cudaMemcpy2D(host_matrices[2],
               width * sizeof(float),
               device_matrices[2],
               pitch,
               width * sizeof(float),
               height,
               cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  float max_error = 0.0f;
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      float sum = host_matrices[0][i * width + j] + host_matrices[1][i * width + j];
      max_error = fmax(max_error, fabs(host_matrices[2][i * width + j] - sum));
    }
  }
  float total_time = 0;

  cudaEventElapsedTime(&total_time, start, stop);

  out << height << " " << width << " " << threads_count_1 << " " << threads_count_2 << " " << max_error << " "
      << total_time << std::endl;

  FreeDeviceMemory(device_matrices);
  FreeHostMemory(host_matrices);
}

int main() {
  std::ofstream results;
  results.open("task3");

  for (size_t height = 1 << 2; height <= (1 << 10); height <<= 2) {
    for (size_t width = 1 << 2; width <= (1 << 10); width <<= 2) {
      for (size_t thread_count_1 = 4; thread_count_1 <= 32; thread_count_1 *= 2) {
        Run(height, width, thread_count_1, thread_count_1, results);
      }
    }
  }

  results.close();

}
