// circle --verbose -O3 -std=c++20 -I../ubu-tot -sm_80 device_inclusive_scan.cpp -lcudart -lfmt -o device_inclusive_scan
#include "measure_bandwidth_of_invocation.hpp"
#include "validate_result.hpp"
#include <fmt/core.h>
#include <cub/cub.cuh>
#include <optional>
#include <numeric>
#include <stdexcept>
#include <ubu/ubu.hpp>
#include <vector>

template<ubu::sized_vector_like I, ubu::sized_vector_like R, class BinaryFunction>
ubu::cuda::event device_inclusive_scan(ubu::cuda::device_executor gpu, void* temporary_storage, std::size_t temporary_storage_size, const ubu::cuda::event& before, I input, R result, BinaryFunction op)
{
  if(cudaError_t e = cudaStreamWaitEvent(gpu.stream(), before.native_handle()))
  {
    throw std::runtime_error(fmt::format("CUDA error after cudaStreamWaitEvent: {}", cudaGetErrorString(e)));
  }

  // sm_80: 60 registers / 403.013 GB/s ~ 92% peak bandwidth on RTX 3070
  if(cudaError_t e = cub::DeviceScan::InclusiveScan(temporary_storage, temporary_storage_size, input.data(), result.data(), op, input.size(), gpu.stream()))
  {
    throw std::runtime_error(fmt::format("CUDA error after cub::DeviceScan::InclusiveScan: {}", cudaGetErrorString(e)));
  }

  return {gpu.device(), gpu.stream()};
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

template<ubu::sized_vector_like V, class BinaryFunction, class I>
  requires std::is_trivially_copyable_v<V>
std::vector<ubu::tensor_element_t<V>> sequential_inclusive_scan(V input, BinaryFunction op, I init)
{
  using namespace std;
  using namespace ubu;

  using T = tensor_element_t<V>;

  auto h_input = to_host(input);

  vector<T> result;
  result.reserve(h_input.size());

  // this loop is inclusive_scan
  optional<T> carry_in = init;
  for(auto x : h_input)
  {
    if(carry_in)
    {
      result.push_back(op(*carry_in, x));
    }
    else
    {
      result.push_back(x);
    }

    carry_in = result.back();
  }

  return result;
}

void test_device_inclusive_scan(std::size_t n)
{
  using namespace std;
  using namespace ubu::cuda;

  device_vector<int> input(n, 1);
  std::iota(input.begin(), input.end(), 0);
  device_vector<int> result(n);
  std::plus op;

  device_executor ex;

  // compute temporary storage requirements
  std::size_t temp_storage_size = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_storage_size, input.data(), result.data(), op, n);
  device_vector<std::byte> temporary_storage(temp_storage_size);

  event before = ubu::initial_happening(ex);

  // compute the result on the GPU
  device_inclusive_scan(ex, temporary_storage.data(), temp_storage_size, before, std::span(input), std::span(result), op).wait();

  // compute the expected result on the CPU
  vector<int> expected = sequential_inclusive_scan(std::span(input), op, 0);

  // check the result
  validate_result(expected, to_host(input), to_host(result), fmt::format("test_device_inclusive_scan({})", n));
}

void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_device_inclusive_scan(" << sz << ")..." << std::flush;
    }

    test_device_inclusive_scan(sz);

    if(verbose)
    {
      std::cout << "OK" << std::endl;
    }
  }
}

double test_performance(std::size_t size, std::size_t num_trials)
{
  int init = 13;
  device_vector<int> input(size);
  device_vector<int> result(size);

  std::span input_view(input);
  std::span result_view(result);
  std::plus op;

  ubu::cuda::device_executor ex;

  // compute temporary storage requirements
  std::size_t temp_storage_size = 0;
  cub::DeviceScan::InclusiveScan(nullptr, temp_storage_size, input_view.data(), result_view.data(), op, size);
  device_vector<std::byte> temporary_storage(temp_storage_size);

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  device_inclusive_scan(ex, temporary_storage.data(), temp_storage_size, before, input_view, result_view, op);

  std::size_t num_bytes = input_view.size_bytes() + result_view.size_bytes();

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, num_bytes, [&]
  {
    device_inclusive_scan(ex, temporary_storage.data(), temp_storage_size, before, input_view, result_view, op);
  });
}

double theoretical_peak_bandwidth_in_gigabytes_per_second()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  double memory_clock_mhz = static_cast<double>(prop.memoryClockRate) / 1000.0;
  double memory_bus_width_bits = static_cast<double>(prop.memoryBusWidth);

  return (memory_clock_mhz * memory_bus_width_bits * 2 / 8.0) / 1024.0;
}

constexpr double performance_regression_threshold_as_percentage_of_peak_bandwidth = 0.9;

int main(int argc, char** argv)
{
  std::size_t performance_size = ubu::cuda::device_allocator<int>().max_size() / 3;
  std::size_t num_performance_trials = 1000;
  std::size_t correctness_size = performance_size;

  if(argc == 2)
  {
    std::string_view arg(argv[1]);
    if(arg != "quick")
    {
      std::cerr << "Unrecognized argument \"" << arg << "\"" << std::endl;
      return -1;
    }

    correctness_size = 1 << 16;
    performance_size /= 10;
    num_performance_trials = 30;
  }

  std::size_t max_size = 23456789;

  std::cout << "Testing correctness... " << std::flush;
  test_correctness(correctness_size, correctness_size > 23456789);
  std::cout << "Done." << std::endl;
  
  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(performance_size, num_performance_trials);
  std::cout << "Done." << std::endl;

  double peak_bandwidth = theoretical_peak_bandwidth_in_gigabytes_per_second();
  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
  std::cout << "Percent peak bandwidth: " << bandwidth / peak_bandwidth << "%" << std::endl;

  if(bandwidth / peak_bandwidth < performance_regression_threshold_as_percentage_of_peak_bandwidth)
  {
    std::cerr << "Theoretical peak bandwidth: " << peak_bandwidth << " GB/s " << std::endl;
    std::cerr << "Regression detected." << std::endl;
    return -1;
  }

  std::cout << "OK" << std::endl;

  return 0;
}

