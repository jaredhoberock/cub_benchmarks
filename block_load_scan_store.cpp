// circle --verbose -O3 -std=c++20 -I../ubu-tot -sm_60 block_load_scan_store.cpp -lcudart -lfmt -o block_load_scan_store
#include "measure_bandwidth_of_invocation.hpp"
#include "validate_result.hpp"
#include <fmt/core.h>
#include <cub/cub.cuh>
#include <numeric>
#include <span>
#include <ubu/ubu.hpp>

template<ubu::span_like I, ubu::span_like R, class Function>
ubu::cuda::event load_scan_store_after(ubu::cuda::device_executor gpu, const ubu::cuda::event& before, I input, R result, Function op)
{
  using namespace ubu;

  size_t num_elements = input.size();
  constexpr size_t max_num_elements_per_thread = 11;
  constexpr size_t block_size = 128;
  constexpr size_t max_tile_size = max_num_elements_per_thread * block_size;
  auto num_blocks = ceil_div(num_elements, max_tile_size);

  // kernel configuration
  std::pair kernel_shape(block_size, num_blocks);

  // 40 registers / 405.372 GB/s ~ 0.93% peak bandwidth
  return bulk_execute_after(gpu,
                            before,
                            kernel_shape,
                            [=](ubu::int2 idx)
  {
    using BlockLoad = cub::BlockLoad<int, 128, 11, cub::BLOCK_LOAD_TRANSPOSE>;
    using BlockScan = cub::BlockScan<int, 128, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockStore = cub::BlockStore<int, 128, 11, cub::BLOCK_STORE_TRANSPOSE>;

    __shared__ union
    {
      typename BlockLoad::TempStorage load;
      typename BlockScan::TempStorage scan;
      typename BlockStore::TempStorage store;
    } temp_storage;

    // The block's range of elements to consume
    size_t block_begin = blockIdx.x * max_tile_size;
    size_t block_end = std::min(num_elements, block_begin + max_tile_size);

    // Load a segment of consecutive items that are blocked across threads
    int thread_data[11];
    BlockLoad(temp_storage.load).Load(input.data() + block_begin, thread_data, block_end - block_begin);
    __syncthreads();

    // Scan across the block
    BlockScan(temp_storage.scan).InclusiveScan(thread_data, thread_data, op);
    __syncthreads();

    // Store the items to the result
    BlockStore(temp_storage.store).Store(result.data() + block_begin, thread_data, block_end - block_begin);
    __syncthreads();
  });
}

// postcondition: is_injective(result)
constexpr ubu::layout_of_rank<3> auto layout_for_scan(std::size_t n)
{
  using namespace ubu;

  auto num_elements_per_thread = 11_c;
  auto num_threads_per_block = 128_c;
  auto tile_size = num_elements_per_thread * num_threads_per_block;
  auto num_blocks = ceil_div(n, tile_size);

  std::tuple shape(num_elements_per_thread, num_threads_per_block, num_blocks);

  return strided_layout(shape);
}

template<ubu::sized_vector_like I, class BinaryFunction>
  requires std::is_trivially_copyable_v<I>
std::vector<ubu::tensor_element_t<I>> sequential_inclusive_scan_tiles(I input, BinaryFunction op)
{
  using namespace std;
  using namespace ubu;

  using T = tensor_element_t<I>;

  auto h_input = to_host(input);
  auto tiles = compose(std::span(h_input), layout_for_scan(h_input.size()));

  vector<T> result;
  result.reserve(h_input.size());

  auto [max_num_elements_per_thread, block_size, num_blocks] = shape(tiles);
  auto tile_size = max_num_elements_per_thread * block_size;

  // loop over blocks
  for(int block_idx = 0; block_idx != num_blocks; ++block_idx)
  {
    std::optional<T> carry_in;

    auto tile = slice(tiles, tuple(_, _, block_idx));

    vector<T> result_tile;

    // this loop is inclusive_scan
    for(auto x : tile)
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

    result.insert(result.end(), result_tile.begin(), result_tile.end());
  }

  return result;
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test_load_scan_store_after(std::size_t n)
{
  device_vector<int> input(n, 1);
  std::iota(input.begin(), input.end(), 0);
  device_vector<int> result(n);

  ubu::cuda::device_executor ex;
  ubu::cuda::event before = ubu::initial_happening(ex);

  // compute the result on the GPU
  load_scan_store_after(ex, before, std::span(input), std::span(result), std::plus()).wait();

  // compute the expectded result on the CPU
  std::vector<int> expected = sequential_inclusive_scan_tiles(std::span(input), std::plus());

  // check the result
  validate_result(expected, to_host(input), to_host(result), fmt::format("load_scan_store_after({})", n));
}

void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_load_scan_store_after(" << sz << ")...";
    }

    test_load_scan_store_after(sz);

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

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  load_scan_store_after(ex, before, input_view, result_view, std::plus{});

  std::size_t num_bytes = input_view.size_bytes() + result_view.size_bytes();

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, num_bytes, [&]
  {
    load_scan_store_after(ex, before, input_view, result_view, std::plus{});
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

constexpr double performance_regression_threshold_as_percentage_of_peak_bandwidth = 0.92;
constexpr double performance_progression_threshold_as_percentage_of_peak_bandwidth = 0.94;

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

  std::cout << "Testing correctness... " << std::flush;
  test_correctness(correctness_size, correctness_size > 23456789);
  std::cout << "Done." << std::endl;
  
  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(performance_size, num_performance_trials);
  std::cout << "Done." << std::endl;

  double peak_bandwidth = theoretical_peak_bandwidth_in_gigabytes_per_second();
  std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

  double pct_peak_bandwidth = bandwidth / peak_bandwidth;
  std::cout << "Percent peak bandwidth: " << pct_peak_bandwidth << "%" << std::endl;

  if(pct_peak_bandwidth < performance_regression_threshold_as_percentage_of_peak_bandwidth)
  {
    std::cerr << "Theoretical peak bandwidth: " << peak_bandwidth << " GB/s " << std::endl;
    std::cerr << "Regression detected." << std::endl;
    return -1;
  }
  else if(pct_peak_bandwidth > performance_progression_threshold_as_percentage_of_peak_bandwidth)
  {
    std::cerr << "Theoretical peak bandwidth: " << peak_bandwidth << " GB/s " << std::endl;
    std::cerr << "Progression detected." << std::endl;
    return -1;
  }

  std::cout << "OK" << std::endl;

  return 0;
}

