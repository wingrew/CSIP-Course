#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// CPU Naive GEMM implementation
// C = alpha * A * B + beta * C
// A: M x K, B: K x N, C: M x N
void cpu_gemm_naive(const float* A, const float* B, float* C, 
                   int M, int N, int K, float alpha, float beta) {
}

// GPU Naive GEMM kernel - each thread computes one output element
__global__ void gpu_gemm_naive_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K, float alpha, float beta) {
}

// GPU Naive GEMM wrapper
void gpu_gemm_naive(const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K, float alpha, float beta) {
    // Use 16x16 thread blocks
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    gpu_gemm_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// GPU opt GEMM wrapper
void gpu_gemm_opt(const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K, float alpha, float beta) {
    // 请给出分块及调用方法

}




// cuBLAS GEMM wrapper for comparison
void cublas_gemm(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C,
                int M, int N, int K, float alpha, float beta) {
    // cuBLAS uses column-major order, but our matrices are in row-major order
    // To compute C = alpha * A * B + beta * C in row-major:
    // We use the identity: (A*B)^T = B^T * A^T
    // Since row-major A is equivalent to column-major A^T, we compute:
    // C^T = alpha * B^T * A^T + beta * C^T, then interpret result as row-major C
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B, N,  // B^T in column-major (our row-major B)
                            d_A, K,  // A^T in column-major (our row-major A)  
                            &beta,
                            d_C, N)); // C^T in column-major (our row-major C)
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Initialize matrix with random values
void initialize_matrix(std::vector<float>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

// Check if two matrices are approximately equal
bool check_correctness(const std::vector<float>& C1, const std::vector<float>& C2, 
                      int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; i++) {
        if (std::abs(C1[i] - C2[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << C1[i] << " vs " << C2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Calculate GFLOPS
double calculate_gflops(int M, int N, int K, double time_ms) {
    // GEMM operations: 2*M*N*K floating point operations
    double flops = 2.0 * M * N * K;
    double gflops = flops / (time_ms * 1e6); // Convert ms to seconds, then to GFLOPS
    return gflops;
}

// Performance benchmark function
void benchmark_gemm(int M, int N, int K, int num_iterations = 10) {
    std::cout << "\n=== GEMM Benchmark: M=" << M << ", N=" << N << ", K=" << K << " ===" << std::endl;
    
    // Initialize matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_gpu_naive(M * N, 0.0f);
    std::vector<float> h_C_cublas(M * N, 0.0f);
    
    initialize_matrix(h_A, M * K);
    initialize_matrix(h_B, K * N);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Benchmark CPU Naive
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        std::fill(h_C_cpu.begin(), h_C_cpu.end(), 0.0f);
        cpu_gemm_naive(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K, alpha, beta);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;
    double cpu_gflops = calculate_gflops(M, N, K, cpu_time);
    
    // Benchmark GPU Naive
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        gpu_gemm_naive(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    end = std::chrono::high_resolution_clock::now();
    double gpu_naive_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;
    double gpu_naive_gflops = calculate_gflops(M, N, K, gpu_naive_time);
    
    // Copy result back for correctness check
    CUDA_CHECK(cudaMemcpy(h_C_gpu_naive.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Benchmark cuBLAS
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        cublas_gemm(handle, d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    end = std::chrono::high_resolution_clock::now();
    double cublas_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;
    double cublas_gflops = calculate_gflops(M, N, K, cublas_time);
    
    // Copy result back for correctness check
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Naive:     " << std::setw(8) << cpu_time << " ms, " 
              << std::setw(8) << cpu_gflops << " GFLOPS" << std::endl;
    std::cout << "GPU Naive:     " << std::setw(8) << gpu_naive_time << " ms, " 
              << std::setw(8) << gpu_naive_gflops << " GFLOPS (Speedup: " 
              << cpu_time / gpu_naive_time << "x)" << std::endl;
    std::cout << "cuBLAS:        " << std::setw(8) << cublas_time << " ms, " 
              << std::setw(8) << cublas_gflops << " GFLOPS (Speedup: " 
              << cpu_time / cublas_time << "x)" << std::endl;
    
    // Check correctness
    bool cpu_gpu_match = check_correctness(h_C_cpu, h_C_gpu_naive, M * N);
    bool cpu_cublas_match = check_correctness(h_C_cpu, h_C_cublas, M * N);
    
    std::cout << "Correctness - CPU vs GPU Naive: " << (cpu_gpu_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "Correctness - CPU vs cuBLAS:    " << (cpu_cublas_match ? "PASS" : "FAIL") << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
}

int main() {
    // Print GPU information
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    
    // Run benchmarks for different matrix sizes
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096}
    };
    
    for (auto& size : test_sizes) {
        int M, N, K;
        std::tie(M, N, K) = size;
        benchmark_gemm(M, N, K);
    }
    
    return 0;
}