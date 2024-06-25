#include <rocwmma/rocwmma.hpp>

#include "precision.h"

using namespace rocwmma;


template<>
class __align__(4) fragment<matrix_a, 16, 16, 32, precision::fp8, row_major> {
  public:
    const static size_t num_elements = 2;
    VRegF32x2 regs;
};

template<>
class __align__(4) fragment<matrix_b, 16, 16, 32, precision::fp8, col_major> {
  public:
    const static size_t num_elements = 2;
    VRegF32x2 regs;
};

template<typename MatrixT, typename DataLayout>
__device__ void fill_fragment(fragment<MatrixT, 16, 16, 32, precision::fp8, DataLayout>& frag, float value) {
  for (size_t i = 0; i < frag.num_elements; i++) {
    frag.regs.data[i] = value;
  }
}

// MFMA compiler intrinsic syntax:
// https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/#mfma-compiler-intrinsic-syntax
inline __device__ void mma_sync_llvm(
  fragment<accumulator, 16, 16, 32, float>& d,
  const fragment<matrix_a, 16, 16, 32, precision::fp8, row_major>& a,
  const fragment<matrix_b, 16, 16, 32, precision::fp8, col_major>& b,
  const fragment<accumulator, 16, 16, 32, float>& c) {

  const size_t M = 16;
  const size_t N = 16;
  const size_t K = 32;
  const size_t output_elements_per_thread = M * N / __AMDGCN_WAVEFRONT_SIZE__;

  size_t tid_x = threadIdx.x % __AMDGCN_WAVEFRONT_SIZE__;
  size_t tid_y = threadIdx.x / __AMDGCN_WAVEFRONT_SIZE__;

  size_t mk = tid_y + K * tix_x;
  size_t kn = tid_x + N * tid_y;

  long amk = a.regs.data[mk];
  long bkn = b.regs.data[kn];
  VecT<float, 4> dmn;
  VecT<float, 4> cmn; // todo: what to put for C here?
#if __gfx940__ || __gfx941__ || __gfx942__
  dmn.data = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(amk, bkn, cmn.data, 0, 0, 0);
#endif

  for (size_t i = 0; i < output_elements_per_thread; i++) {
    const int idx = tid_x + i * N + tid_y * output_elements_per_thread * N;
    (*d).data[idx] = dmn.data[i];
  }
}
