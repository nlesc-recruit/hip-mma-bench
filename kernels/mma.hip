#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include "precision.h"

using namespace rocwmma;

#define REPEAT_COUNT 32768

#define START                                        \
  fragment<accumulator, M, N, K, Tout> sum;          \
  fragment<matrix_a, M, N, K, Tmma, row_major> aFrag; \
  fragment<matrix_b, M, N, K, Tmma, col_major> bFrag; \
  fill_fragment(sum, static_cast<Tout>(0));          \
  fill_fragment(aFrag, static_cast<Tin>(0));         \
  fill_fragment(bFrag, static_cast<Tin>(0));         \
  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
#define END                               \
  }                                       \
  Tout* ptr = &data[threadIdx.y * M * N]; \
  store_matrix_sync(ptr, sum, N, mem_row_major);

template <typename Tin, typename Tmma, typename Tout, unsigned M, unsigned N, unsigned K>
__device__ void mma_kernel(Tout* data) {
  START
  mma_sync(sum, aFrag, bFrag, sum);
  END
}

#include "mma_m16n16k32_fp32fp8fp8fp32.h"

template <typename Tin, typename Tmma, typename Tout, unsigned M, unsigned N, unsigned K>
__device__ void mma_kernel_llvm(Tout* data) {
  START
  mma_sync_llvm(sum, aFrag, bFrag, sum);
  END
}

__global__ void mma_fp8_16_16_32(void* data) {
  mma_kernel_llvm<float, precision::fp8, float, 16, 16, 32>((float*)data);
}

__global__ void mma_s8_16_16_16(void* data) {
  mma_kernel<signed char, signed char, int, 16, 16, 16>((int*)data);
}

__global__ void mma_f16_16_16_16(void* data) {
  mma_kernel<half, half, float, 16, 16, 16>((float*)data);
}

__global__ void mma_bf16_16_16_16(void* data) {
  mma_kernel<hip_bfloat16, hip_bfloat16, float, 16, 16, 16>((float*)data);
}

__global__ void mma_f32_16_16_16(void* data) {
 mma_kernel<float, float, float, 16, 16, 16>((float*)data);
}

__global__ void mma_f64_16_16_16(void* data) {
 mma_kernel<double, double, double, 16, 16, 16>((double*)data);
}
