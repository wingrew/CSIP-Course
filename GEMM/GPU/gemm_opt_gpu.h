#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

__global__ void gpu_gemm_opt_kernel(
    float * __restrict__ a,
    float * __restrict__ b,
    float * __restrict__ c,
    const int M,
    const int N,
    const int K,
    float alpha, 
    float beta
);

