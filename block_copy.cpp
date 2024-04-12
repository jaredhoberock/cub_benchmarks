// circle --verbose -O3 -std=c++20 -I. -sm_60 block_copy.cpp -lcudart -lfmt -o block_copy
#include "measure_bandwidth_of_invocation.hpp"
#include "validate_result.hpp"
#include <fmt/core.h>
#include <cub/cub.cuh>
#include <numeric>
#include <span>
#include <ubu/ubu.hpp>

template<ubu::span_like I, ubu::span_like R>
ubu::cuda::event copy_after(ubu::cuda::device_executor gpu, const ubu::cuda::event& before, I input, R result)
{
  using namespace ubu;

  size_t num_elements = input.size();
  constexpr size_t max_num_elements_per_thread = 11;
  constexpr size_t block_size = 128;
  constexpr size_t max_tile_size = max_num_elements_per_thread * block_size;
  auto num_blocks = ceil_div(num_elements, max_tile_size);

  // kernel configuration
  std::pair kernel_shape(block_size, num_blocks);

  // 46 registers / 391.858 GB/s ~ 89.7% peak bandwidth
  return bulk_execute_after(gpu,
                            before,
                            kernel_shape,
                            [=](ubu::int2 idx)
  {
    using BlockLoad = cub::BlockLoad<int, 128, 11, cub::BLOCK_LOAD_TRANSPOSE>;
    using BlockStore = cub::BlockStore<int, 128, 11, cub::BLOCK_STORE_TRANSPOSE>;

    __shared__ union
    {
      typename BlockLoad::TempStorage load;
      typename BlockStore::TempStorage store;
    } temp_storage;

    // The block's range of elements to consume
    size_t block_begin = blockIdx.x * max_tile_size;
    size_t block_end = std::min(num_elements, block_begin + max_tile_size);

    // Load a segment of consecutive items that are blocked across threads
    int thread_data[max_num_elements_per_thread];
    BlockLoad(temp_storage.load).Load(input.data() + block_begin, thread_data, block_end - block_begin);
    __syncthreads();

    // Store the items to the result
    BlockStore(temp_storage.store).Store(result.data() + block_begin, thread_data, block_end - block_begin);
    __syncthreads();
  });
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test_copy_after(std::size_t n)
{
  device_vector<int> input(n, 1);
  std::iota(input.begin(), input.end(), 0);
  device_vector<int> result(n);

  ubu::cuda::device_executor ex;
  ubu::cuda::event before = ubu::initial_happening(ex);

  // compute the result on the GPU
  copy_after(ex, before, std::span(input), std::span(result)).wait();

  // check the result
  validate_result(to_host(input), to_host(input), to_host(result), fmt::format("copy_after({})", n));
}

void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_copy_after(" << sz << ")...";
    }

    test_copy_after(sz);

    if(verbose)
    {
      std::cout << "OK" << std::endl;
    }
  }
}

double test_performance(std::size_t size, std::size_t num_trials)
{
  device_vector<int> input(size);
  device_vector<int> result(size);

  std::span input_view(input);
  std::span result_view(result);

  ubu::cuda::device_executor ex;
  ubu::cuda::device_allocator<std::byte> alloc;

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  copy_after(ex, before, input_view, result_view);

  std::size_t num_bytes = input_view.size_bytes() + result_view.size_bytes();

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, num_bytes, [&]
  {
    copy_after(ex, before, input_view, result_view);
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

int main(int argc, char** argv)
{
  if(argc == 2)
  {
    std::string_view arg(argv[1]);
    if(arg != "quick")
    {
      std::cerr << "Unrecognized argument \"" << arg << "\"" << std::endl;
      return -1;
    }

    test_correctness(1 << 16);
    std::cout << "OK" << std::endl;
    return 0;
  }

  std::size_t max_size = 23456789;

  std::cout << "Testing correctness... " << std::flush;
  test_correctness(max_size, max_size > 23456789);
  std::cout << "Done." << std::endl;
  
  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(ubu::cuda::device_allocator<int>().max_size() / 2, 1000);
  std::cout << "Done." << std::endl;

  double peak_bandwidth = theoretical_peak_bandwidth_in_gigabytes_per_second();
  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
  std::cout << "Percent peak bandwidth: " << bandwidth / peak_bandwidth << "%" << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

