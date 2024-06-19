#include <cmath>
#include <thread>

#include <cxxopts.hpp>

#include "common.h"

#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT hipDeviceAttributeMultiprocessorCount
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE hipDeviceAttributeClockRate
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK hipDeviceAttributeMaxThreadsPerBlock
#define CUdeviceptr hipDeviceptr_t

static constexpr int w1 = 20;
static constexpr int w2 = 7;

void print_ops(double gops, measurement& m) {
  const double seconds = m.runtime * 1e-3;
  if (gops != 0) {
    cout << ", " << setw(w2) << gops / seconds * 1e-3 << " TOps/s";
  }
}

void print_power(double gops, measurement& m) {
  if (m.power > 1) {
    cout << ", " << setw(w2) << m.power << " W";
  }
}

void print_efficiency(double gops, measurement& m) {
  const double seconds = m.runtime * 1e-3;
  const double power = m.power;
  if (gops != 0 && power > 1) {
    cout << ", " << setw(w2) << gops / seconds / power << " GOps/W";
  }
}

void print_bandwidth(double gbytes, measurement& m) {
  const double seconds = m.runtime * 1e-3;
  if (gbytes != 0) {
    cout << ", " << setw(w2) << gbytes / seconds << " GB/s";
  }
}

void print_oi(double gops, double gbytes) {
  if (gops != 0 && gbytes != 0) {
    float operational_intensity = gops / gbytes;
    cout << ", " << setw(w2) << operational_intensity << " Op/byte";
  }
}

void report(string name, double gops, double gbytes, measurement& m) {
  const double milliseconds = m.runtime;
  const double seconds = milliseconds * 1e-3;
  cout << setw(w1) << string(name) << ": ";
  cout << setprecision(2) << fixed;
  cout << setw(w2) << milliseconds << " ms";
  print_ops(gops, m);
  print_power(gops, m);
  print_efficiency(gops, m);
  print_bandwidth(gbytes, m);
  print_oi(gops, gbytes);
  cout << endl;
}

unsigned roundToPowOf2(unsigned number) {
  double logd = log(number) / log(2);
  logd = floor(logd);

  return (unsigned)pow(2, (int)logd);
}

cxxopts::Options setupCommandLineParser(const char* argv[]) {
  cxxopts::Options options(argv[0], "Benchmark for BeamFormerKernel");

  const unsigned NR_BENCHMARKS = 1;
  const unsigned NR_ITERATIONS = 1;
#if defined(HAVE_PMT)
  const unsigned MEASURE_POWER = false;
  const unsigned BENCHMARK_DURATION = 4000;  // ms
#endif
  const unsigned DEVICE_ID = 0;

  options.add_options()(
      "nr_benchmarks", "Number of benchmarks",
      cxxopts::value<unsigned>()->default_value(std::to_string(NR_BENCHMARKS)))(
      "nr_iterations", "Number of kernel iteration per benchmark",
      cxxopts::value<unsigned>()->default_value(std::to_string(NR_ITERATIONS)))(
#if defined(HAVE_PMT)
      "measure_power", "Measure power",
      cxxopts::value<bool>()->default_value(std::to_string(MEASURE_POWER)))(
      "benchmark_duration", "Approximate number of ms to run the benchmark",
      cxxopts::value<unsigned>()->default_value(
          std::to_string(BENCHMARK_DURATION)))(
#endif
      "device_id", "Device ID",
      cxxopts::value<unsigned>()->default_value(std::to_string(DEVICE_ID)))(
      "h,help", "Print help");

  return options;
}

cxxopts::ParseResult getCommandLineOptions(int argc, const char* argv[]) {
  cxxopts::Options options = setupCommandLineParser(argv);

  try {
    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(EXIT_SUCCESS);
    }

    return result;

  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "Error parsing command-line options: " << e.what()
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

Benchmark::Benchmark(int argc, const char* argv[]) {
  // Parse command-line options
  cxxopts::ParseResult results = getCommandLineOptions(argc, argv);
  const unsigned device_number = results["device_id"].as<unsigned>();
  nr_benchmarks_ = results["nr_benchmarks"].as<unsigned>();
  nr_iterations_ = results["nr_iterations"].as<unsigned>();
#if defined(HAVE_PMT)
  measure_power_ = results["measure_power"].as<bool>();
  benchmark_duration_ = results["benchmark_duration"].as<unsigned>();
#endif

  // Setup HIP
  cu::init();
  device_ = std::make_unique<cu::Device>(device_number);
  context_ =
      std::make_unique<cu::Context>(CU_CTX_SCHED_BLOCKING_SYNC, *device_);
  stream_ = std::make_unique<cu::Stream>();

  // Print HIP device information
  std::cout << "Device " << device_number << ": " << device_->getName();
  std::cout << " (" << multiProcessorCount();
  if (isGfx9())
    std::cout << "CUs, ";
  else if (isGfx11())
    std::cout << "WGPs, ";
  else
    std::cout << "units, ";
  std::cout << clockRate() * 1e-6 << " Ghz)" << std::endl;

#if defined(HAVE_PMT)
  pm_ = std::move(pmt::Create("rocm"));
#endif
}

// isGfx9 and isGfx11 based on
// https://github.com/ROCm/rocWMMA/blob/develop/samples/common.hpp
bool Benchmark::isGfx9() {
    hipDeviceProp_t mProps;
    hipGetDeviceProperties(&mProps, *device_);

    std::string deviceName(mProps.gcnArchName);

    return ((deviceName.find("gfx908") != std::string::npos)
            || (deviceName.find("gfx90a") != std::string::npos)
            || (deviceName.find("gfx940") != std::string::npos)
            || (deviceName.find("gfx941") != std::string::npos)
            || (deviceName.find("gfx942") != std::string::npos));
}

bool Benchmark::isGfx11() {
    hipDeviceProp_t mProps;
    hipGetDeviceProperties(&mProps, *device_);

    std::string deviceName(mProps.gcnArchName);

    return ((deviceName.find("gfx1100") != std::string::npos)
            || (deviceName.find("gfx1101") != std::string::npos)
            || (deviceName.find("gfx1102") != std::string::npos));
}

void Benchmark::allocate(size_t bytes) {
  cu::HostMemory h_data(bytes);
  d_data_ = std::make_unique<cu::DeviceMemory>(bytes);
  std::memset(h_data, 1, bytes);
  stream_->memcpyHtoDAsync(*d_data_, h_data, bytes);
  stream_->synchronize();
}

void Benchmark::run(void* kernel, dim3 grid, dim3 block, const char* name,
                    double gops, double gbytes) {
  measurement measurement = run_kernel(kernel, grid, block);
  report(name, gops, gbytes, measurement);
}

int Benchmark::multiProcessorCount() {
  return device_->getAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
}

int Benchmark::clockRate() {
  return device_->getAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
}

int Benchmark::maxThreadsPerBlock() {
  return device_->getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

size_t Benchmark::totalGlobalMem() { return context_->getTotalMemory(); }

void launch_kernel(void* kernel, cu::Stream& stream, dim3 grid, dim3 block,
                   void* data);

measurement Benchmark::run_kernel(void* kernel, dim3 grid, dim3 block) {
  cu::Event event_start;
  cu::Event event_end;
  void* data = reinterpret_cast<void*>(static_cast<CUdeviceptr>(*d_data_));

// Benchmark with power measurement
#if defined(HAVE_PMT)
  if (measurePower()) {
    float milliseconds = 0;
    unsigned nr_iterations = 0;

    std::thread thread([&] {
      stream_->record(event_start);
      launch_kernel(kernel, *stream_, grid, block, data);
      stream_->record(event_end);
      event_end.synchronize();
      milliseconds = event_end.elapsedTime(event_start);
      nr_iterations = benchmarkDuration() / milliseconds;
      stream_->record(event_start);
      for (int i = 0; i < nr_iterations; i++) {
        launch_kernel(kernel, *stream_, grid, block, data);
      }
      stream_->record(event_end);
      event_end.synchronize();
      milliseconds = event_end.elapsedTime(event_start);
    });
    std::this_thread::sleep_for(
        std::chrono::milliseconds(int(0.5 * benchmarkDuration())));
    pmt::State state_start = pm_->Read();
    std::this_thread::sleep_for(
        std::chrono::milliseconds(int(0.2 * benchmarkDuration())));
    pmt::State state_end = pm_->Read();
    if (thread.joinable()) {
      thread.join();
    }

    measurement measurement;
    measurement.runtime = milliseconds / nr_iterations;
    measurement.power = pmt::PMT::watts(state_start, state_end);

    return measurement;
  }
#endif

  // Benchmark (timing only)
  stream_->record(event_start);
  for (int i = 0; i < nrIterations(); i++) {
    launch_kernel(kernel, *stream_, grid, block, data);
  }
  stream_->record(event_end);
  event_end.synchronize();
  const float milliseconds = event_end.elapsedTime(event_start);
  measurement measurement;
  measurement.runtime = milliseconds / nrIterations();
  measurement.power = 0;
  return measurement;
}
