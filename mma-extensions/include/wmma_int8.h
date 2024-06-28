#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

/*
./matrix_calculator.py -a RDNA3 -i v_wmma_i32_16x16x16_iu8 -d
Architecture: RDNA3
Instruction: V_WMMA_I32_16X16X16_IU8
    Encoding: VOP3P
    VOP3P Opcode: 0x44
    Matrix Dimensions:
        M: 16
        N: 16
        K: 16
    Execution statistics:
        Ops: 8192
        Execution cycles: 32
        Ops/WGP/cycle: 1024
        Can co-execute with VALU: False
    Wave32 register usage:
        GPRs required for A: 4
        GPRs required for B: 4
        GPRs required for C: 8
        GPRs required for D: 8
        GPR alignment requirement: 4 bytes
    Wave64 register usage:
        GPRs required for A: 4
        GPRs required for B: 4
        GPRs required for C: 4
        GPRs required for D: 4
        GPR alignment requirement: 4 bytes
    VOP3P register encoding:
        A matrix source field: Src0
        B matrix source field: Src1
        C matrix source field: Src2
        D matrix source field: Vdst
    Register data types:
        Src0: IU8 (Signed/unsigned 8-bit integer)
        Src1: IU8 (Signed/unsigned 8-bit integer)
        Src2: int32 (Signed 32-bit integer)
        Vdst: int32 (Signed 32-bit integer)
    Register modifiers:
        OPSEL[1:0] supported: False
        OPSEL[2] supported: False
        NEG bits supported: True
    Matrix element to register mapping with no modifiers:
        A[i][k] GPR: floor(k / 4).[8*(k % 4)+7 : 8*(k % 4)]
        A[i][k] Lane: i and i+16. Also i+32 and i+48 in wave64.
        B[k][j] GPR: floor(k / 4).[8*(k % 4)+7 : 8*(k % 4)]
        B[k][j] Lane: j and j+16. Also j+32 and j+48 in wave64.
        C or D[i][j] GPR: floor((16 * i) / wave_width)
        C or D[i][j] Lane: ((16 * i) % wave_width) + j
    Register to matrix element mapping with no modifiers:
        A i: (lane % 16)
        A k: 4 * GPR_num + floor(GPR_bits / 8)
        B j: (lane % 16)
        B k: 4 * GPR_num + floor(GPR_bits / 8)
        C or D i: (wave_width / 16) * GPR_num + floor(lane / 16)
        C or D j: (lane % 16)
*/

// We are reimplementing existing features in rocwmma with this int8 example, so we cannot specialize the existing templates
// Therefore we declare all these empty templates instead

template<typename MatrixT, unsigned M, unsigned N, unsigned K, typename DataT, typename DataLayout=void>
class __align__(4) custom_fragment {};

template<typename MatrixT, unsigned M, unsigned N, unsigned K, typename DataT, typename DataLayout>
__device__ void custom_fill_fragment();

template<typename MatrixT, unsigned M, unsigned N, unsigned K, typename DataT, typename DataLayout>
__device__ void custom_load_matrix_sync();

template<typename MatrixT, unsigned M, unsigned N, unsigned K, typename DataT, typename DataLayout>
__device__ void custom_store_matrix_sync();

template<typename MatrixT, unsigned M, unsigned N, unsigned K, typename DataT, typename DataLayout>
inline __device__ void custom_mma_sync();


template<>
class __align__(4) custom_fragment<matrix_a, 16, 16, 16, signed char, row_major> {
    public:
        const static size_t num_elements = 4;
        VRegI32x4 regs;
};

template<>
class __align__(4) custom_fragment<matrix_b, 16, 16, 16, signed char, col_major> {
    public:
        const static size_t num_elements = 4;
        VRegI32x4 regs;
};

template<>
class __align__(4) custom_fragment<accumulator, 16, 16, 16, int> {
    public:
        const static size_t num_elements = 8;
        AccRegI32x8 regs;
};

template<typename MatrixT, typename DataT, typename DataLayout>
__device__ void custom_fill_fragment(custom_fragment<MatrixT, 16, 16, 16, DataT, DataLayout>& frag, const signed char value) {
    for (size_t i = 0; i < frag.num_elements; i++) {
        frag.regs.data[i] = static_cast<DataT>(value);
    }
}

template<typename MatrixT, typename DataT, typename DataLayout>
__device__ void custom_fill_fragment(custom_fragment<MatrixT, 16, 16, 16, DataT, DataLayout>& frag, const int value) {
    for (size_t i = 0; i < frag.num_elements; i++) {
        frag.regs.data[i] = static_cast<DataT>(value);
    }
}

__device__ void custom_load_matrix_sync(custom_fragment<matrix_a, 16, 16, 16, signed char, row_major>& frag, const signed char* ptr, unsigned lda) {
    // A i: (lane % 16)
    // A k: 4 * GPR_num + floor(GPR_bits / 8)
    unsigned i = threadIdx.x % 16;
    const unsigned elements_per_register = 4;  // 4 chars in one 32-bit register
    for (size_t reg = 0; reg < frag.num_elements; reg++) {
        frag.regs.data[reg] = 0;
        for (size_t ele = 0; ele < elements_per_register; ele++) {
            unsigned k = 4 * reg + ele;
            signed char value = ptr[i * lda + k];
            frag.regs.data[reg] |= value << (8 * ele);
        }
    }
}

__device__ void custom_load_matrix_sync(custom_fragment<matrix_b, 16, 16, 16, signed char, col_major>& frag, const signed char* ptr, unsigned lda) {
    // B j: (lane % 16)
    // B k: 4 * GPR_num + floor(GPR_bits / 8)
    unsigned j = threadIdx.x % 16;
    const unsigned elements_per_register = 4;  // 4 chars in one 32-bit register
    for (size_t reg = 0; reg < frag.num_elements; reg++) {
        frag.regs.data[reg] = 0;
        for (size_t ele = 0; ele < elements_per_register; ele++) {
            unsigned k = 4 * reg + ele;
            signed char value = ptr[j * lda + k];
            frag.regs.data[reg] |= value << (8 * ele);
        }
    }
}


template<typename MatrixT>
__device__ void custom_store_matrix_sync(int *ptr, const custom_fragment<MatrixT, 16, 16, 16, int>& frag, const unsigned lda, const layout_t layout) {
    // C or D i: (wave_width / 16) * GPR_num + floor(lane / 16)
    // C or D j: (lane % 16)
    unsigned j = threadIdx.x % 16;
    for (size_t reg = 0; reg < frag.num_elements; reg++) {
        unsigned i = (__AMDGCN_WAVEFRONT_SIZE__ / 16) * reg + (threadIdx.x / 16);
        if (layout == mem_row_major) {
            ptr[i * lda + j] = frag.regs.data[reg];
        } else {
            ptr[j * lda + i] = frag.regs.data[reg];
        }

    }
}

inline __device__ void custom_mma_sync(
    custom_fragment<accumulator, 16, 16, 16, int>& d,
    const custom_fragment<matrix_a, 16, 16, 16, signed char, row_major>& a,
    const custom_fragment<matrix_b, 16, 16, 16, signed char, col_major>& b,
    const custom_fragment<accumulator, 16, 16, 16, int>& c
) {

    enum : uint32_t {
        UNSIGNED = 0,
        SIGNED = 1
    };

#if __gfx1100__ || __gfx1101__ || __gfx1102__
    d.regs.data = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(SIGNED, a.regs.data, SIGNED, b.regs.data, c.regs.data, SIGNED);
#endif
}