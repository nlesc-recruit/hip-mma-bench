#include <limits.h>

#include <iostream>
#include <functional>
#include <random>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cudawrappers/cu.hpp>

#include "wmma_int8.h"

using namespace rocwmma;

inline void hip_check(hipError_t err) {
    if (err != hipSuccess) {
        std::cout << "ERROR: " << hipGetErrorString(err) << std::endl;
    }
}

template<unsigned M, unsigned N, unsigned K, unsigned NLOOP=1>
__global__ void int8_rocwmma(const signed char *a, const signed char *b, int *c) {
    fragment<matrix_a, M, N, K, signed char, row_major> frag_a;
    fragment<matrix_b, M, N, K, signed char, col_major> frag_b;
    fragment<accumulator, M, N, K, int> frag_c;

    fill_fragment(frag_c, 0);
    load_matrix_sync(frag_a, a, K);
    load_matrix_sync(frag_b, b, K);
    for (unsigned i=0; i < NLOOP; i++) {
        mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    store_matrix_sync(c, frag_c, N, mem_row_major);
}

template<unsigned M, unsigned N, unsigned K, unsigned NLOOP=1>
__global__ void int8_custom(const signed char *a, const signed char *b, int *c) {
    custom_fragment<matrix_a, M, N, K, signed char, row_major> frag_a;
    custom_fragment<matrix_b, M, N, K, signed char, col_major> frag_b;
    custom_fragment<accumulator, M, N, K, int> frag_c;

    custom_fill_fragment(frag_c, 0);
    custom_load_matrix_sync(frag_a, a, K);
    custom_load_matrix_sync(frag_b, b, K);
    for (unsigned i=0; i < NLOOP; i++) {
        custom_mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    custom_store_matrix_sync(c, frag_c, N, mem_row_major);
}



int main() {
    cu::init();
    cu::Device device(0);
    cu::Stream stream;
    const unsigned wave_size = device.getAttribute(hipDeviceAttributeWarpSize);
    const unsigned multiProcessorCount = device.getAttribute(hipDeviceAttributeMultiprocessorCount);
    const unsigned M = 16;
    const unsigned N = 16;
    const unsigned K = 16;

    const unsigned bytes_a = M * K * sizeof(signed char);
    const unsigned bytes_b = N * K * sizeof(signed char);
    const unsigned bytes_c = M * N * sizeof(int);


    cu::HostMemory a(bytes_a);
    cu::HostMemory b(bytes_b);

    // note: while the data type is signed, the RDNA3 WMMA INT8 instruction work on unsigned numbers so values <0 are not supported
    auto generator = std::bind(std::uniform_int_distribution<>(0, CHAR_MAX), std::default_random_engine());
    for (int i = 0; i < M * K; i++) {
        static_cast<signed char *>(a)[i] = generator();
    }

    for (int i = 0; i < N * K; i++) {
        static_cast<signed char *>(b)[i] = generator();
    }

    signed char *d_a;
    signed char *d_b;
    int *d_c;

    hip_check(hipMalloc(&d_a, bytes_a));
    hip_check(hipMalloc(&d_b, bytes_b));
    hip_check(hipMalloc(&d_c, bytes_c));

    // rocWMMA impl
    stream.memcpyHtoDAsync(d_a, a, bytes_a);
    stream.memcpyHtoDAsync(d_b, b, bytes_b);
    hip_check(hipMemset(d_c, 0, bytes_c));

    dim3 threads(wave_size, 1, 1);
    dim3 grid(1, 1, 1);
    int8_rocwmma<M, N, K><<<grid, threads, 0, stream>>>(d_a, d_b, d_c);
    hip_check(hipGetLastError());

    cu::HostMemory c_rocwmma(bytes_c);
    stream.memcpyDtoHAsync(c_rocwmma, d_c, bytes_c);

    // custom impl
    stream.memcpyHtoDAsync(d_a, a, bytes_a);
    stream.memcpyHtoDAsync(d_b, b, bytes_b);
    hip_check(hipMemset(d_c, 0, bytes_c));

    int8_custom<M, N, K><<<grid, threads, 0, stream>>>(d_a, d_b, d_c);
    hip_check(hipGetLastError());

    cu::HostMemory c_custom(bytes_c);
    stream.memcpyDtoHAsync(c_custom, d_c, bytes_c);

    // verification
    stream.synchronize();

    unsigned errs = 0;
    for (int i=0; i < M * N; i++) {
        const int& c_ref = static_cast<int *>(c_rocwmma)[i];
        const int& c_val = static_cast<int *>(c_custom)[i];
        if (c_ref != c_val) {
            unsigned n = i % N;
            unsigned m = i / N;
            std::cout << "Failed at M, N = ("<< m << ", " << n << "). Found " << c_val << ", expected " << c_ref << std::endl;
            if (++errs >= 10) {
                break;
            }
        }
    }
    if (errs == 0) {
        std::cout << "Result ok" << std::endl;
    }


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

    // rocWMMA
    hip_check(hipEventRecord(start));
    for (int i = 0; i < niter; i++) {
        int8_rocwmma<16, 16, 16, nloop><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
    }
    hip_check(hipEventRecord(stop));
    hip_check(hipEventSynchronize(stop));
    hip_check(hipDeviceSynchronize());
    hip_check(hipEventElapsedTime(&time, start, stop));
    time /= niter;
    tflops = 2ULL * 1e-9 * M * N * K * nblock * nloop * nwave_per_block / time;
    std::cout << "rocWMMA:" << std::endl;
    std::cout << "Time per loop: " << time << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;

    // custom
    hip_check(hipEventRecord(start));
    for (int i = 0; i < niter; i++) {
        int8_custom<16, 16, 16, nloop><<<grid_bench, threads_bench, 0, stream>>>(d_a, d_b, d_c);
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