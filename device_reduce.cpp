// circle --verbose -O3 -std=c++20 -I../ubu-tot -sm_80 device_reduce.cpp -lcudart -lfmt -o device_reduce
#include "measure_bandwidth_of_invocation.hpp"
#include "validate.hpp"
#include <fmt/core.h>
#include <cub/cub.cuh>
#include <optional>
#include <numeric>
#include <stdexcept>
#include <ubu/ubu.hpp>
#include <vector>

template<ubu::sized_vector_like I, std::random_access_iterator R, class T, class BinaryFunction>
ubu::cuda::event device_reduce(ubu::cuda::device_executor gpu, void* temporary_storage, std::size_t temporary_storage_size, const ubu::cuda::event& before, I input, R result, T init, BinaryFunction op)
{
  if(cudaError_t e = cudaStreamWaitEvent(gpu.stream(), before.native_handle()))
  {
    throw std::runtime_error(fmt::format("CUDA error after cudaStreamWaitEvent: {}", cudaGetErrorString(e)));
  }

  // sm_80: 31 registers / 712.302 GB/s ~ 95% peak bandwidth on RTX A5000
  if(cudaError_t e = cub::DeviceReduce::Reduce(temporary_storage, temporary_storage_size, input.data(), result, input.size(), op, init, gpu.stream()))
  {
    throw std::runtime_error(fmt::format("CUDA error after cub::DeviceReduce::Reduce: {}", cudaGetErrorString(e)));
  }

  return {gpu.device(), gpu.stream()};
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test_device_reduce(std::size_t n)
{
  using namespace std;
  using namespace ubu::cuda;

  device_vector<int> input(n, 1);
  std::iota(input.begin(), input.end(), 0);
  device_vector<int> result(1, 0);
  std::plus op;
  int init = 13;

  device_executor ex;

  // compute temporary storage requirements
  std::size_t temp_storage_size = 0;
  cub::DeviceReduce::Reduce(nullptr, temp_storage_size, input.data(), result.data(), n, op, init);
  device_vector<std::byte> temporary_storage(temp_storage_size);

  event before = ubu::initial_happening(ex);

  // compute the result on the GPU
  device_reduce(ex, temporary_storage.data(), temp_storage_size, before, std::span(input), result.data(), init, op).wait();

  // compute the expected result on the CPU
  auto h_input = to_host(input);
  int expected = std::accumulate(input.begin(), input.end(), init, op);

  // check the result
  validate_result(expected, result[0], fmt::format("test_device_reduce({})", n));
}

void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_device_reduce(" << sz << ")..." << std::flush;
    }

    test_device_reduce(sz);

    if(verbose)
    {
      std::cout << "OK" << std::endl;
    }
  }
}

double test_performance(std::size_t size, std::size_t num_trials)
{
  device_vector<int> input(size);
  device_vector<int> result(1);

  std::span input_view(input);
  std::span result_view(result);
  int init = 13;
  std::plus op;

  ubu::cuda::device_executor ex;

  // compute temporary storage requirements
  std::size_t temp_storage_size = 0;
  cub::DeviceReduce::Reduce(nullptr, temp_storage_size, input_view.data(), result.data(), size, op, init);
  device_vector<std::byte> temporary_storage(temp_storage_size);

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  device_reduce(ex, temporary_storage.data(), temp_storage_size, before, input_view, result.data(), init, op);

  std::size_t num_bytes = input_view.size_bytes() + result_view.size_bytes();

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, num_bytes, [&]
  {
    device_reduce(ex, temporary_storage.data(), temp_storage_size, before, input_view, result.data(), init, op);
  });
}

performance_expectations_t device_reduce_expectations = {
  {"NVIDIA GeForce RTX 3070", {0.94, 0.96}},
  {"NVIDIA RTX A5000", {0.94, 0.96}}
};

int main(int argc, char** argv)
{
  std::size_t performance_size = choose_large_problem_size_using_heuristic<int>(1);
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

  std::cout << "Testing correctness... " << std::flush;
  test_correctness(correctness_size, correctness_size > 23456789);
  std::cout << "Done." << std::endl;
  
  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(performance_size, num_performance_trials);
  std::cout << "Done." << std::endl;

  report_performance(bandwidth_to_performance(bandwidth), device_reduce_expectations);

  std::cout << "OK" << std::endl;

  return 0; 
}

