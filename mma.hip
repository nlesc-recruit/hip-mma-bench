#include "common.h"

__global__ void mma_s8_16_16_16(void* ptr);
__global__ void mma_f16_16_16_16(void* ptr);
__global__ void mma_bf16_16_16_16(void* ptr);
__global__ void mma_f32_16_16_16(void* ptr);
__global__ void mma_f64_16_16_16(void* ptr);

int main(int argc, const char* argv[]) {
  Benchmark benchmark(argc, argv);

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();
  unsigned warp_size = benchmark.warpSize();

  // Kernel dimensions
  int nr_thread_blocks = multiProcessorCount * 512;
  int nr_warps_per_thread_block = 4;
  dim3 grid(nr_thread_blocks);
  dim3 block(warp_size, nr_warps_per_thread_block);

  size_t sizeof_data = nr_warps_per_thread_block * 16 * 16 * sizeof(double);

  // Amount of work performed
  int nr_iterations = 32768;
  const double gops =
      1e-9 * nr_iterations * nr_warps_per_thread_block * nr_thread_blocks;
  const double gbytes = 0;

  benchmark.allocate(sizeof_data);

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void*>(&mma_s8_16_16_16), grid, block,
                  "mma_s8_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma_f16_16_16_16), grid, block,
                  "mma_f16_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma_bf16_16_16_16), grid, block,
                  "mma_bf16_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
    // F32 and F64 are only available on CDNA (gfx9)
    if (benchmark.isGfx9()) {
        benchmark.run(reinterpret_cast<void*>(&mma_f32_16_16_16 ), grid, block,
                      "mma_f32_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
        benchmark.run(reinterpret_cast<void*>(&mma_f64_16_16_16 ), grid, block,
                      "mma_f64_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
    }
  }

  return EXIT_SUCCESS;
}
