// SpeciaLUT benchmark for CUDA kernels: a specialized kernel vs one kernel
// that branches on its run-time options.
//
// The options are the same for all the threads, so the run-time kernel has no
// warp divergence to lose time on. What the specialization can still win is
// what the compiler makes of the constants: no branch instructions, no code
// and no registers for the paths not taken, loops of a known length.
//
// As in benchmark.cpp, both kernels are built from the very same source and
// their results are compared.

#include "specialut.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

enum class Kernel { poly, spiky, rational, count };

struct DensityAtRuntime {
  Kernel kernel;
  int dim;
  bool weighted;
  bool limited;
};

template <Kernel K, int Dim, bool Weighted, bool Limited>
struct DensityAtCompileTime {
  static constexpr Kernel kernel = K;
  static constexpr int dim = Dim;
  static constexpr bool weighted = Weighted;
  static constexpr bool limited = Limited;
};

/// Sum of kernel weights over the neighbours of particle i, which are the
/// particles up to `reach` places away
template <typename Options>
__device__ __forceinline__ float
density_at(Options o, const float *x, const float *y, const float *z,
           const float *mass, int i, int n, float inv_h2) {
  constexpr int reach = 16;
  float sum = 0.0f;
  for (int k = -reach; k <= reach; ++k) {
    const int j = min(max(i + k, 0), n - 1);

    // distance in 2 or 3 dimensions
    const float dx = x[i] - x[j];
    const float dy = y[i] - y[j];
    float r2 = dx * dx + dy * dy;
    if (o.dim == 3) {
      const float dz = z[i] - z[j];
      r2 += dz * dz;
    }
    const float q2 = r2 * inv_h2;

    // smoothing kernel
    float w = 0.0f;
    switch (o.kernel) {
    case Kernel::poly: {
      const float t = fmaxf(0.0f, 1.0f - q2);
      w = t * t * t;
    } break;
    case Kernel::spiky: {
      const float t = fmaxf(0.0f, 1.0f - sqrtf(q2));
      w = t * t * t;
    } break;
    default: {
      const float t = 1.0f + q2;
      w = 1.0f / (t * t);
    } break;
    }

    if (o.weighted) {
      w *= mass[j];
    }
    sum += w;
  }
  return o.limited ? fminf(sum, 4.0f) : sum;
}

__global__ void density_runtime(DensityAtRuntime o, const float *x,
                                const float *y, const float *z,
                                const float *mass, float *out, int n,
                                float inv_h2) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    out[i] = density_at(o, x, y, z, mass, i, n, inv_h2);
  }
}

template <Kernel K, int Dim, bool Weighted, bool Limited>
__global__ void density_special(const float *x, const float *y, const float *z,
                                const float *mass, float *out, int n,
                                float inv_h2) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    out[i] = density_at(DensityAtCompileTime<K, Dim, Weighted, Limited>{}, x, y,
                        z, mass, i, n, inv_h2);
  }
}

constexpr int n = 1 << 22;
constexpr int launches = 20;
constexpr int runs = 9;
constexpr unsigned block = 256;

/// Time of `launches` kernels, shortest of several runs, in milliseconds
template <typename F> double milliseconds(F &&launch) {
  double best = 1e300;
  for (int r = 0; r < runs; ++r) {
    cudaDeviceSynchronize();
    const auto start = std::chrono::steady_clock::now();
    for (int l = 0; l < launches; ++l) {
      launch();
    }
    cudaDeviceSynchronize();
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start;
    best = std::min(best, elapsed.count());
  }
  return best;
}

float *to_device(std::vector<float> const &host) {
  float *device = nullptr;
  if (cudaMalloc(&device, host.size() * sizeof(float)) != cudaSuccess) {
    return nullptr;
  }
  cudaMemcpy(device, host.data(), host.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  return device;
}

bool same(std::vector<float> const &a, std::vector<float> const &b) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(std::abs(a[i] - b[i]) <= 1e-4f * std::max(1.0f, std::abs(a[i])))) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  std::vector<float> hx(n), hy(n), hz(n), hmass(n), result(n), reference(n);
  for (int i = 0; i < n; ++i) {
    const float t = float(i);
    hx[i] = 0.05f * float(i % 4096);
    hy[i] = 0.2f * std::sin(0.37f * t);
    hz[i] = 0.2f * std::cos(0.11f * t);
    hmass[i] = 1.0f + 0.25f * std::sin(0.05f * t);
  }
  const float *x = to_device(hx);
  const float *y = to_device(hy);
  const float *z = to_device(hz);
  const float *mass = to_device(hmass);
  float *out = to_device(result);
  if (!x || !y || !z || !mass || !out) {
    std::puts("No usable CUDA device, nothing was measured.");
    return 77;
  }

  cudaDeviceProp device{};
  cudaGetDeviceProperties(&device, 0);
  std::printf("SpeciaLUT CUDA benchmark on %s\n", device.name);
  std::printf("%d particles, %d launches, best of %d runs each\n\n", n,
              launches, runs);
  std::printf("  %-34s %12s %12s %8s\n", "options", "run-time", "SpeciaLUT",
              "speedup");

  const dim3 grid((n + block - 1) / block);
  const float inv_h2 = 1.0f / (0.5f * 0.5f);

  // 3 kernels x 2 dimensions x 2 x 2 flags = 24 specializations
  SpeciaLUT::CudaChooser<TABULATE(density_special), Kernel::count,
                         SpeciaLUT::values<2, 3>, 2, 2>
      chooser;
  chooser.prepare(grid, block);

  const char *kernels[] = {"poly", "spiky", "rational"};
  int mismatches = 0;
  double total_runtime = 0, total_lut = 0;
  for (int k = 0; k < 3; ++k) {
    for (int dim = 2; dim <= 3; ++dim) {
      for (int flags = 0; flags < 2; ++flags) { // both off, both on
        const DensityAtRuntime o{static_cast<Kernel>(k), dim, flags != 0,
                                 flags != 0};

        const double runtime_ms = milliseconds([&] {
          density_runtime<<<grid, block>>>(o, x, y, z, mass, out, n, inv_h2);
        });
        cudaMemcpy(reference.data(), out, n * sizeof(float),
                   cudaMemcpyDeviceToHost);

        const auto specialized =
            chooser(o.kernel, o.dim, o.weighted, o.limited); // chosen once
        const double lut_ms =
            milliseconds([&] { specialized(x, y, z, mass, out, n, inv_h2); });
        cudaMemcpy(result.data(), out, n * sizeof(float),
                   cudaMemcpyDeviceToHost);

        const bool agree =
            cudaGetLastError() == cudaSuccess && same(reference, result);
        mismatches += !agree;
        total_runtime += runtime_ms;
        total_lut += lut_ms;

        char options[64];
        std::snprintf(options, sizeof(options), "%-8s %dD %s", kernels[k], dim,
                      flags ? "weighted, limited" : "plain");
        std::printf("  %-34s %9.2f ms %9.2f ms %7.2fx%s\n", options, runtime_ms,
                    lut_ms, runtime_ms / lut_ms,
                    agree ? "" : "   RESULTS DIFFER");
      }
    }
  }
  std::printf("  %-34s %9.2f ms %9.2f ms %7.2fx\n", "all of the above",
              total_runtime, total_lut, total_runtime / total_lut);

  std::printf("\nspeedup = run-time branching / SpeciaLUT; above 1 the "
              "specialization is faster.\n");
  if (mismatches != 0) {
    std::printf("%d case(s) computed different results!\n", mismatches);
    return 1;
  }
  return 0;
}
