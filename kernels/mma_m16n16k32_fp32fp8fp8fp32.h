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
__device__ void fill_fragment(fragment<MatrixT, 16, 16, 32, precision::fp8, DataLayout>& frag, const char value) {
    for (size_t i = 0; i < frag.num_elements; i++) {
        frag.regs.data[i] = value | (value << 8) | (value << 16) | (value << 24);
    }
}

// MFMA compiler intrinsic syntax:
// https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/#mfma-compiler-intrinsic-syntax
inline __device__ void mma_sync_llvm(
  fragment<accumulator, 16, 16, 32, float>& d,
  const fragment<matrix_a, 16, 16, 32, precision::fp8, row_major>& a,
  const fragment<matrix_b, 16, 16, 32, precision::fp8, col_major>& b,
  const fragment<accumulator, 16, 16, 32, float>& c) {

#if __gfx940__ || __gfx941__ || __gfx942__
  (*d).data = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(((VRegI64x1 const&)(a.regs)).data[0], ((VRegI64x1 const&)(b.regs)).data[0], (*c).data, 0, 0, 0);
#endif
}
