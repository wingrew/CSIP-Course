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

void cpu_gemm_opt(const float* A, const float* B, float* C, 
                   int M, int N, int K, float alpha, float beta) {
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
    std::vector<float> h_C_cpu_opt(M * N, 0.0f);
    
    initialize_matrix(h_A, M * K);
    initialize_matrix(h_B, K * N);
    
    float alpha = 1.0f, beta = 0.0f;

    
    // Benchmark CPU Naive
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        std::fill(h_C_cpu.begin(), h_C_cpu.end(), 0.0f);
        cpu_gemm_naive(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K, alpha, beta);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;
    double cpu_gflops = calculate_gflops(M, N, K, cpu_time);


    // Benchmark CPU opt
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        std::fill(h_C_cpu_opt.begin(), h_C_cpu_opt.end(), 0.0f);
        cpu_gemm_opt(h_A.data(), h_B.data(), h_C_cpu_opt.data(), M, N, K, alpha, beta);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_opt_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;
    double cpu_opt_gflops = calculate_gflops(M, N, K, cpu_opt_time);
    
    
    // Print results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Naive:     " << std::setw(8) << cpu_time << " ms, " 
              << std::setw(8) << cpu_gflops << " GFLOPS" << std::endl;
    std::cout << "CPU Opt:     " << std::setw(8) << cpu_opt_time << " ms, " 
              << std::setw(8) << cpu_opt_gflops << " GFLOPS" << std::endl;

    bool cpu_opt_match = check_correctness(h_C_cpu, h_C_cpu_opt, M * N);
    
    std::cout << "Correctness - CPU vs GPU Naive: " << (cpu_opt_match ? "PASS" : "FAIL") << std::endl;
    
}

int main() {

    
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