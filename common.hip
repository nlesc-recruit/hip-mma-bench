#include <cudawrappers/cu.hpp>

void launch_kernel(void* kernel, cu::Stream& stream, dim3 grid, dim3 block,
                   void* data) {
  ((void (*)(void*))kernel)<<<grid, block, 0, stream>>>(data);
}
