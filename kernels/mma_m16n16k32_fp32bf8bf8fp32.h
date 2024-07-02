#include <rocwmma/rocwmma.hpp>

#include "precision.h"

using namespace rocwmma;

template<>
class __align__(4) fragment<matrix_a, 16, 16, 32, precision::bf8, row_major> {
  public:
    const static size_t num_elements = 2;
    VRegF32x2 regs;
};

template<>
class __align__(4) fragment<matrix_b, 16, 16, 32, precision::bf8, col_major> {
  public:
    const static size_t num_elements = 2;
    VRegF32x2 regs;
};

template<typename MatrixT, typename DataLayout>
__device__ void fill_fragment(fragment<MatrixT, 16, 16, 32, precision::bf8, DataLayout>& frag, const char value) {
    for (size_t i = 0; i < frag.num_elements; i++) {
        frag.regs.data[i] = value | (value << 8) | (value << 16) | (value << 24);
    }
}

__device__ void load_matrix_sync(fragment<matrix_a, 16, 16, 32, precision::bf8, row_major>& frag, const char* ptr, unsigned ldm) {
    // A i: (lane % 16)
    // A k: 8 * floor(lane / 16) + 4 * GPR_num + floor(GPR_bits / 8)
    const unsigned lane = threadIdx.x % __AMDGCN_WAVEFRONT_SIZE__;
    const unsigned i = lane % 16;
    const unsigned epr = 4;  // 4 elements per register (4 chars in one 32-bit register)
    for (size_t reg = 0; reg < frag.num_elements; reg++) {
        frag.regs.data[reg] = 0;
        int tmp = 0;
        for (size_t ele = 0; ele < epr; ele++) {
            unsigned k = frag.num_elements * epr * (lane / 16) + epr * reg + ele;
            char value = ptr[i * ldm + k];
            tmp |= value << (8 * ele);
        }
        frag.regs.data[reg] = *reinterpret_cast<float *>(&tmp);
    }
}

__device__ void load_matrix_sync(fragment<matrix_b, 16, 16, 32, precision::bf8, col_major>& frag, const char* ptr, unsigned ldm) {
    // B j: (lane % 16)
    // B k: 8 * floor(lane / 16) + 4 * GPR_num + floor(GPR_bits / 8)
    const unsigned lane = threadIdx.x % __AMDGCN_WAVEFRONT_SIZE__;
    const unsigned j = lane % 16;
    const unsigned epr = 4;  // 4 elements per register (4 chars in one 32-bit register)s
    for (size_t reg = 0; reg < frag.num_elements; reg++) {
        frag.regs.data[reg] = 0;
        int tmp = 0;
        for (size_t ele = 0; ele < epr; ele++) {
            unsigned k = frag.num_elements * epr * (lane / 16) + epr * reg + ele;
            char value = ptr[j * ldm + k];
            tmp |= value << (8 * ele);
        }
        frag.regs.data[reg] = *reinterpret_cast<float *>(&tmp);
    }
}

// MFMA compiler intrinsic syntax:
// https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/#mfma-compiler-intrinsic-syntax
inline __device__ void mma_sync_llvm(
  fragment<accumulator, 16, 16, 32, float>& d,
  const fragment<matrix_a, 16, 16, 32, precision::bf8, row_major>& a,
  const fragment<matrix_b, 16, 16, 32, precision::bf8, col_major>& b,
  const fragment<accumulator, 16, 16, 32, float>& c) {

#if __gfx940__ || __gfx941__ || __gfx942__
  (*d).data = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(((VRegI64x1 const&)(a.regs)).data[0], ((VRegI64x1 const&)(b.regs)).data[0], (*c).data, 0, 0, 0);
#endif
}
