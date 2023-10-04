#pragma once


__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result);
