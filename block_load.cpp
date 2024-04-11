// circle --verbose -O3 -std=c++20 -I../ubu-tot -sm_60 block_load.cpp -lcudart -o block_load -lfmt
#include "measure_bandwidth_of_invocation.hpp"
#include <fmt/core.h>
#include <cub/cub.cuh>
#include <vector>
#include <ubu/ubu.hpp>

template<ubu::span_like I>
ubu::cuda::event load_after(ubu::cuda::device_executor gpu, const ubu::cuda::event& before, I input)
{
  using namespace ubu;

  size_t num_elements = input.size();
  constexpr size_t max_num_elements_per_thread = 15;
  constexpr size_t block_size = 128;
  constexpr size_t max_tile_size = max_num_elements_per_thread * block_size;
  auto num_blocks = ceil_div(num_elements, max_tile_size);

  // kernel configuration
  std::pair kernel_shape(block_size, num_blocks);

  // 32 registers / 409.363 GB/s
  return bulk_execute_after(gpu,
                            before,
                            kernel_shape,
                            [=](ubu::int2 idx)
  {
    using BlockLoad = cub::BlockLoad<int, block_size, max_num_elements_per_thread, cub::BLOCK_LOAD_TRANSPOSE>;
    __shared__ typename BlockLoad::TempStorage temp_storage;

    // The block's range of elements to consume
    size_t block_begin = blockIdx.x * max_tile_size;
    size_t block_end = std::min(num_elements, block_begin + max_tile_size);

    // Load a segment of consecutive items that are blocked across threads
    int thread_data[max_num_elements_per_thread];
    BlockLoad(temp_storage).Load(input.data() + block_begin, thread_data, block_end - block_begin);
    __syncthreads();
  });
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

double test_performance(std::size_t size, std::size_t num_trials)
{
  device_vector<int> input(size);
  std::span input_view(input);
  ubu::cuda::device_executor ex;
  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  load_after(ex, before, input_view);

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, input_view.size_bytes(), [&]
  {
    load_after(ex, before, input_view);
  });
}

int main(int argc, char** argv)
{
  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(ubu::cuda::device_allocator<int>().max_size() / 2, 1000);
  std::cout << "Done." << std::endl;

  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

  std::cout << "OK" << std::endl;

  return 0;
}

