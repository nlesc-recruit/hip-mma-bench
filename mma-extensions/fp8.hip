#include <limits.h>

#include <iostream>
#include <fstream>
#include <functional>
#include <random>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cudawrappers/cu.hpp>

#include "mfma_fp8.h"

using namespace rocwmma;

inline void hip_check(hipError_t err) {
    if (err != hipSuccess) {
        std::cout << "ERROR: " << hipGetErrorString(err) << std::endl;
    }
}

template<unsigned M, unsigned N, unsigned K, unsigned NLOOP=1>
__global__ void fp8_custom(const char *a, const char *b, float *c) {
    fragment<matrix_a, M, N, K, precision::fp8, row_major> frag_a;
    fragment<matrix_b, M, N, K, precision::fp8, col_major> frag_b;
    fragment<accumulator, M, N, K, float> frag_c;

    fill_fragment(frag_c, 0.0f);
    load_matrix_sync(frag_a, a, K);
    load_matrix_sync(frag_b, b, K);
    for (unsigned i=0; i < NLOOP; i++) {
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    store_matrix_sync(c, frag_c, N, mem_row_major);
}



int main() {
    cu::init();
    cu::Device device(0);
    cu::Stream stream;
    const unsigned wave_size = device.getAttribute(hipDeviceAttributeWarpSize);
    const unsigned multiProcessorCount = device.getAttribute(hipDeviceAttributeMultiprocessorCount);
    const unsigned M = 16;
    const unsigned N = 16;
    const unsigned K = 32;

    const unsigned bytes_a = M * K * sizeof(char);
    const unsigned bytes_b = N * K * sizeof(char);
    const unsigned bytes_c = M * N * sizeof(float);


    cu::HostMemory a(bytes_a);
    cu::HostMemory b(bytes_b);

    auto generator = std::bind(std::uniform_int_distribution<>(SCHAR_MIN, SCHAR_MAX), std::default_random_engine());
    for (int i = 0; i < M * K; i++) {
        static_cast<char *>(a)[i] = generator();
    }

    for (int i = 0; i < N * K; i++) {
        static_cast<char *>(b)[i] = generator();
    }

    char *d_a;
    char *d_b;
    float *d_c;

    hip_check(hipMalloc(&d_a, bytes_a));
    hip_check(hipMalloc(&d_b, bytes_b));
    hip_check(hipMalloc(&d_c, bytes_c));

    // custom impl
    stream.memcpyHtoDAsync(d_a, a, bytes_a);
    stream.memcpyHtoDAsync(d_b, b, bytes_b);
    hip_check(hipMemset(d_c, 0, bytes_c));

    dim3 threads(wave_size, 1, 1);
    dim3 grid(1, 1, 1);
    fp8_custom<M, N, K><<<grid, threads, 0, stream>>>(d_a, d_b, d_c);
    hip_check(hipGetLastError());

    cu::HostMemory c_custom(bytes_c);
    stream.memcpyDtoHAsync(c_custom, d_c, bytes_c);
    stream.synchronize();


    // write matrices to disk for offline verification
    std::ofstream out("A.bin", std::ios::binary | std::ios::out);
    out.write(static_cast<char *>(a), bytes_a);
    out = std::ofstream("B.bin", std::ios::binary | std::ios::out);
    out.write(static_cast<char *>(b), bytes_b);
    out = std::ofstream("C_custom.bin", std::ios::binary | std::ios::out);
    out.write(static_cast<char *>(c_custom), bytes_c);

    // performance measurement
    const unsigned niter = 10;
    const unsigned nloop = 32768;
    const unsigned nwave_per_block = 4;
    const unsigned nblock = 512 * multiProcessorCount;
    dim3 threads_bench(wave_size, nwave_per_block, 1);
    dim3 grid_bench(nblock, 1, 1);

    hipEvent_t start;
    hipEvent_t stop;
    float time;
    double tflops;

    hip_check(hipEventCreate(&start));
    hip_check(hipEventCreate(&stop));

    // custom
    hip_check(hipEventRecord(start));
    for (int i = 0; i < niter; i++) {
        fp8_custom<M, N, K, nloop><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
    }
    hip_check(hipEventRecord(stop));
    hip_check(hipEventSynchronize(stop));
    hip_check(hipDeviceSynchronize());
    hip_check(hipEventElapsedTime(&time, start, stop));
    time /= niter;
    tflops = 2ULL * 1e-9 * M * N * K * nblock * nloop * nwave_per_block / time;
    std::cout << "Custom:" << std::endl;
    std::cout << "Time per loop: " << time << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;
}