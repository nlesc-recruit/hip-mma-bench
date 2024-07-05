#include <iomanip>
#include <iostream>

#include <cudawrappers/cu.hpp>

#ifndef COMMON_H
#define COMMON_H

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

#if defined(HAVE_FMT)
#include <fmt/fmt.h>
#endif

using namespace std;

unsigned roundToPowOf2(unsigned number);

typedef struct {
  double runtime; // milliseconds
  double power;   // watts
  int frequency;  // MHz
} measurement;

// Function to report kernel performance
void report(string name, measurement measurement, double gflops = 0,
            double gbytes = 0, double gops = 0);

class Benchmark {
 public:
  Benchmark(int argc, const char* argv[]);

  bool isCDNA();
  bool isCDNA1();
  bool isCDNA2();
  bool isCDNA3();
  bool isRDNA3();
  void allocate(size_t bytes);
  void run(void* kernel, dim3 grid, dim3 block, const char* name,
           double gops = 0, double gbytes = 0);

  int multiProcessorCount();
  int clockRate();
  int maxThreadsPerBlock();
  size_t totalGlobalMem();
  unsigned warpSize();

  unsigned nrBenchmarks() { return nr_benchmarks_; }
  unsigned nrIterations() { return nr_iterations_; }
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  unsigned benchmarkDuration() { return benchmark_duration_; }
#endif
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  bool measureContinuous() {
    bool result = false;
#if defined(HAVE_PMT)
    result |= measure_power_;
#endif
#if defined(HAVE_FMT)
    result |= measure_frequency_;
#endif
    return result;
  }
#endif

 protected:
  measurement run_kernel(void* kernel, dim3 grid, dim3 block);

  unsigned device_number_;
  unsigned nr_benchmarks_;
  unsigned nr_iterations_;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Context> context_;
  std::unique_ptr<cu::Stream> stream_;
  std::unique_ptr<cu::DeviceMemory> d_data_;
#if defined(HAVE_PMT)
  std::shared_ptr<pmt::PMT> pm_;
  bool measure_power_;
#endif
#if defined(HAVE_FMT)
  bool measure_frequency_;
#endif
#if defined(HAVE_FMT) || defined(HAVE_FMT)
  unsigned benchmark_duration_;
#endif
};

#endif  // end COMMON_H
