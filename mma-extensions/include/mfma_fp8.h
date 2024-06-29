#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

/*
./matrix_calculator.py -a CDNA3 -i v_mfma_f32_16x16x32_fp8_fp8 -d
Architecture: CDNA3
Instruction: V_MFMA_F32_16X16X32_FP8_FP8
    Encoding: VOP3P-MAI
    VOP3P Opcode: 0x73
    VOP3P-MAI Opcode: 0x33
    Matrix Dimensions:
        M: 16
        N: 16
        K: 32
        blocks: 1
    Execution statistics:
        FLOPs: 16384
        Execution cycles: 16
        FLOPs/CU/cycle: 4096
        Can co-execute with VALU: True
        VALU co-execution cycles possible: 12
    Register usage:
        GPRs required for A: 2
        GPRs required for B: 2
        GPRs required for C: 4
        GPRs required for D: 4
        GPR alignment requirement: 8 bytes
    VOP3P-MAI register encoding:
        A matrix source field: Src0
        B matrix source field: Src1
        C matrix source field: Src2
        D matrix source field: Vdst
    Register data types:
        Src0: FP8 (AMD 4-bit exponent, 3-bit mantissa floating point)
        Src1: FP8 (AMD 4-bit exponent, 3-bit mantissa floating point)
        Src2: FP32 (IEEE binary32 floating point)
        Vdst: FP32 (IEEE binary32 floating point)
    Register capabilities:
        A matrix can use ArchVGPRs: True
        A matrix can use AccVGPRs: True
        B matrix can use ArchVGPRs: True
        B matrix can use AccVGPRs: True
        C and D matrix can use ArchVGPRs: True
        C and D matrix can use AccVGPRs: True
    Register modifiers:
        Sparse A matrix: False
        CBSZ and ABID bits supported: False
        BLGP bits supported: False
    Matrix element to register mapping with no modifiers:
        A[i][k].block GPR: (floor(k / 4) % 2).[8*(k % 4)+7 : 8*(k % 4)]
        A[i][k].block Lane: 16 * floor(k / 8) + i
        B[k][j].block GPR: (floor(k / 4) % 2).[8*(k % 4)+7 : 8*(k % 4)]
        B[k][j].block Lane: 16 * floor(k / 8) + j
        C or D[i][j].block GPR: (i % 4)
        C or D[i][j].block Lane: 16 * floor(i / 4) + j
    Register to matrix element mapping with no modifiers:
        A i: (lane % 16)
        A k: 8 * floor(lane / 16) + 4 * GPR_num + floor(GPR_bits / 8)
        A block: 0
        B j: (lane % 16)
        B k: 8 * floor(lane / 16) + 4 * GPR_num + floor(GPR_bits / 8)
        B block: 0
        C or D i: 4 * floor(lane / 16) + (GPR_num % 4)
        C or D j: (lane % 16)
        C or D block: 0
*/

namespace precision {
    struct fp8;
}

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

__device__ void load_matrix_sync(fragment<matrix_a, 16, 16, 32, precision::fp8, row_major>& frag, const char* ptr, unsigned lda) {
    // A i: (lane % 16)
    // A k: 8 * floor(lane / 16) + 4 * GPR_num + floor(GPR_bits / 8)
    unsigned i = threadIdx.x % 16;
    const unsigned epr = 4;  // 4 elements per register (4 chars in one 32-bit register)
    for (size_t reg = 0; reg < frag.num_elements; reg++) {
        frag.regs.data[reg] = 0;
        int tmp = 0;
        for (size_t ele = 0; ele < epr; ele++) {
            unsigned k = frag.num_elements * epr * (threadIdx.x / 16) + epr * reg + ele;
            char value = ptr[i * lda + k];
            tmp |= value << (8 * ele);
        }
        frag.regs.data[reg] = *reinterpret_cast<float *>(&tmp);
    }
}

__device__ void load_matrix_sync(fragment<matrix_b, 16, 16, 32, precision::fp8, col_major>& frag, const char* ptr, unsigned lda) {
    // B j: (lane % 16)
    // B k: 8 * floor(lane / 16) + 4 * GPR_num + floor(GPR_bits / 8)
    unsigned j = threadIdx.x % 16;
    const unsigned epr = 4;  // 4 elements per register (4 chars in one 32-bit register)s
    for (size_t reg = 0; reg < frag.num_elements; reg++) {
        frag.regs.data[reg] = 0;
        int tmp = 0;
        for (size_t ele = 0; ele < epr; ele++) {
            unsigned k = frag.num_elements * epr * (threadIdx.x / 16) + epr * reg + ele;
            char value = ptr[j * lda + k];
            tmp |= value << (8 * ele);
        }
        frag.regs.data[reg] = *reinterpret_cast<float *>(&tmp);
    }
}

inline __device__ void mma_sync(
    fragment<accumulator, 16, 16, 32, float>& d,
    const fragment<matrix_a, 16, 16, 32, precision::fp8, row_major>& a,
    const fragment<matrix_b, 16, 16, 32, precision::fp8, col_major>& b,
    const fragment<accumulator, 16, 16, 32, float>& c
) {

    enum : uint32_t {
        UNSIGNED = 0,
        SIGNED = 1
    };

#if __gfx940__ || __gfx941__ || __gfx942__
    (*d).data = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(((VRegI64x1 const&)(a.regs)).data[0], ((VRegI64x1 const&)(b.regs)).data[0], (*c).data, 0, 0, 0);
#endif
}