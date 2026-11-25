// SpeciaLUT tests for CUDA kernels. Every kernel writes into device memory,
// which is copied back and compared with what its specialization must produce.
// The program returns the number of failures.

#include "specialut.hpp"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      ++failures;                                                              \
      std::printf("%s:%d: failed: %s\n", __FILE__, __LINE__, #condition);      \
    }                                                                          \
  } while (false)

enum class Op { add, multiply, count };

/// out[i] = (i op scale) + offset, negated if asked for
template <bool Negate, Op Operation, int Offset>
__global__ void fill(double *out, float scale, unsigned long long n) {
  const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  double v = double(i);
  if constexpr (Operation == Op::add) {
    v += double(scale);
  } else {
    v *= double(scale);
  }
  v += Offset;
  out[i] = Negate ? -v : v;
}

double expected(bool negate, Op op, int offset, float scale, unsigned i) {
  double v = double(i);
  v = op == Op::add ? v + double(scale) : v * double(scale);
  v += offset;
  return negate ? -v : v;
}

template <int Value> __global__ void no_arguments() {}

constexpr unsigned n = 1000;

void test_all_specializations(double *device) {
  SpeciaLUT::CudaChooser<TABULATE(fill), 2, Op::count,
                         SpeciaLUT::values<10, 20>>
      chooser;
  chooser.prepare({(n + 127) / 128, 1, 1}, {128, 1, 1});

  std::vector<double> host(n);
  for (int negate = 0; negate < 2; ++negate) {
    for (Op op : {Op::add, Op::multiply}) {
      for (int offset : {10, 20}) {
        // the arguments are an int and an unsigned, the kernel takes a float
        // and an unsigned long long: they must arrive converted
        CHECK(chooser(negate, op, offset)(device, 3, n) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        CHECK(cudaMemcpy(host.data(), device, n * sizeof(double),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        unsigned wrong = 0;
        for (unsigned i = 0; i < n; ++i) {
          wrong += host[i] != expected(negate != 0, op, offset, 3.0f, i);
        }
        CHECK(wrong == 0);
      }
    }
  }

  // states that are not tabulated are refused, not launched
  bool thrown = false;
  try {
    (void)chooser(true, Op::add, 30);
  } catch (std::out_of_range const &) {
    thrown = true;
  }
  CHECK(thrown);
}

void test_execution_parameters(double *device) {
  SpeciaLUT::CudaChooser<TABULATE(fill), 2, Op::count,
                         SpeciaLUT::values<10, 20>>
      chooser;

  std::vector<double> host(n);

  // nothing prepared: a single block of a single thread, as dim3 defaults to
  CHECK(cudaMemset(device, 0, n * sizeof(double)) == cudaSuccess);
  CHECK(chooser(false, Op::add, 10)(device, 1.0f, n) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(cudaMemcpy(host.data(), device, n * sizeof(double),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(host[0] == expected(false, Op::add, 10, 1.0f, 0));
  CHECK(host[1] == 0.0);

  // execution parameters given for one launch only
  auto kernel = chooser(true, Op::multiply, 20);
  CHECK(kernel.prepare({n, 1, 1}, {1, 1, 1}).launch(device, 2.0f, n) ==
        cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(cudaMemcpy(host.data(), device, n * sizeof(double),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(host[7] == expected(true, Op::multiply, 20, 2.0f, 7));
  CHECK(host[n - 1] == expected(true, Op::multiply, 20, 2.0f, n - 1));

  // and they did not leak back into the chooser
  CHECK(cudaMemset(device, 0, n * sizeof(double)) == cudaSuccess);
  CHECK(chooser(true, Op::multiply, 20)(device, 2.0f, n) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(cudaMemcpy(host.data(), device, n * sizeof(double),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(host[1] == 0.0);
}

/// A kernel that has no specialization for one combination of its parameters
template <bool Negate, int Scale>
  requires(!(Negate && Scale == 20))
__global__ void scaled(double *out, double value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = (Negate ? -1.0 : 1.0) * Scale * value;
  }
}

void test_states_and_rejected_combinations(double *device) {
  // the parameters described once, for the tables of two kernels
  using Options = SpeciaLUT::States<2, SpeciaLUT::values<10, 20>>;
  using Fill = SpeciaLUT::States<2, Op::count, SpeciaLUT::values<10, 20>>;

  Fill::CudaChooser<TABULATE(fill)> fills;
  fills.prepare({(n + 127) / 128, 1, 1}, {128, 1, 1});
  const Fill chosen(true, Op::add, 20); // checked here, once
  CHECK(fills[chosen](device, 3, n) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<double> host(n);
  CHECK(cudaMemcpy(host.data(), device, n * sizeof(double),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  unsigned wrong = 0;
  for (unsigned i = 0; i < n; ++i) {
    wrong += host[i] != expected(true, Op::add, 20, 3.0f, i);
  }
  CHECK(wrong == 0);

  Options::CudaChooser<TABULATE(scaled)> kernels;
  kernels.prepare({1, 1, 1}, {1, 1, 1});
  CHECK(kernels.size() == 3); // 4 combinations, one of them rejected
  CHECK(kernels[Options(false, 20)](device, 1.5) == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(cudaMemcpy(host.data(), device, sizeof(double),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(host[0] == 30.0);

  // the rejected combination has no kernel, and is not launched
  CHECK(kernels.find(true, 20) == nullptr);
  CHECK(kernels.find(true, 10) != nullptr);
  int refused = 0;
  try {
    (void)kernels(true, 20);
  } catch (std::out_of_range const &) {
    ++refused;
  }
  try {
    (void)kernels[Options(true, 20)];
  } catch (std::out_of_range const &) {
    ++refused;
  }
  CHECK(refused == 2);
}

void test_no_arguments() {
  SpeciaLUT::CudaChooser<TABULATE(no_arguments), 3> chooser;
  chooser.prepare({1, 1, 1}, {1, 1, 1});
  CHECK(chooser(2)() == cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
}

} // namespace

int main() {
  double *device = nullptr;
  if (cudaMalloc(&device, n * sizeof(double)) != cudaSuccess) {
    std::puts("No usable CUDA device, nothing was tested.");
    return 77; // reported as skipped
  }

  test_all_specializations(device);
  test_execution_parameters(device);
  test_states_and_rejected_combinations(device);
  test_no_arguments();

  cudaFree(device);

  if (failures == 0) {
    std::puts("All SpeciaLUT CUDA tests passed.");
  } else {
    std::printf("%d check(s) failed.\n", failures);
  }
  return failures == 0 ? 0 : 1;
}
