#include <rocwmma/rocwmma.hpp>

#include "precision.hpp"

using namespace rocwmma;

typedef int _vec2 __attribute__((__vector_size__(2 * sizeof(int))));
typedef int _vec8 __attribute__((__vector_size__(8 * sizeof(int))));

template<>
class __align__(4) fragment<matrix_a, 16, 16, 16, int4_t, row_major> {
public:
    // VRegI32x2 data;
    _vec2 data;
};

template<>
class __align__(4) fragment<matrix_b, 16, 16, 16, int4_t, col_major> {
public:
    // VRegI32x2 data;
    _vec2 data;
};

template<typename MatrixT, typename DataLayout>
__device__ void fill_fragment(fragment<MatrixT, 16, 16, 16, int4_t, DataLayout>& frag, int4_t value) {

}


inline __device__ void mma_sync_llvm(
    fragment<accumulator, 16, 16, 16, int>& d,
    const fragment<matrix_a, 16, 16, 16, int4_t, row_major>&
        a,
    const fragment<matrix_b, 16, 16, 16, int4_t, col_major>&
        b,
    const fragment<accumulator, 16, 16, 16, int>& c) {

    enum : uint32_t {
        UNSIGNED = 0,
        SIGNED = 1
    };

    const _vec8 *c_data = reinterpret_cast<const _vec8*>(&(*c));
    _vec8 *d_data = reinterpret_cast<_vec8*>(&(*d));



#if __gfx1100__ || __gfx1101__ || __gfx1102__
    *d_data = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(SIGNED, a.data, SIGNED, b.data, *c_data, SIGNED);
#endif
}
