// SpeciaLUT benchmarks: LUT dispatch to a specialization vs run-time branching.
//
// Each case runs the very same source twice: once with its options as
// compile-time constants (one specialization per combination, chosen through
// SpeciaLUT), once with the options as ordinary run-time values. The results
// are compared, so a specialization that computes something else is reported.
// Both variants write into the same buffer and their runs are interleaved.
//
// Three cases, on purpose not all in favour of the library:
//   1. a trivial loop with one option: the compiler hoists the branch out of
//      the loop on its own, nothing to gain
//   2. a stencil with four orthogonal options: the branches stay in the inner
//      loop of the run-time version and keep it from being vectorized
//   3. a tiny function chosen for every single element: the wrong way to use
//      a table of function pointers, unless the compiler sees through it

#include "specialut.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#define ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#define ALWAYS_INLINE __forceinline
#else
#define NOINLINE
#define ALWAYS_INLINE inline
#endif

namespace {

/// Hide a value from the optimizer, so that run-time options stay run-time
template <typename T> NOINLINE T opaque(T value) {
  volatile T hidden = value;
  return hidden;
}

constexpr int runs = 9;

template <typename F> double milliseconds(F &&run) {
  const auto start = std::chrono::steady_clock::now();
  run();
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - start;
  return elapsed.count();
}

/// Same values, up to the rounding that differently optimized code is allowed
/// to have (e.g. a multiplication and an addition contracted into one FMA)
bool same(std::vector<float> const &a, std::vector<float> const &b) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(std::abs(a[i] - b[i]) <= 1e-4f * std::max(1.0f, std::abs(a[i])))) {
      return false;
    }
  }
  return true;
}

int mismatches = 0;
double total_runtime_ms = 0;
double total_lut_ms = 0;

void report(const char *options, double runtime_ms, double lut_ms,
            bool same_result) {
  std::printf("  %-34s %9.2f ms %9.2f ms %7.2fx%s\n", options, runtime_ms,
              lut_ms, runtime_ms / lut_ms,
              same_result ? "" : "   RESULTS DIFFER");
  mismatches += !same_result;
}

/// Time both variants, which write their result into the same `out`. The runs
/// are interleaved and the shortest one counts, so that neither variant gets
/// the warmer cache, the better aligned buffer or the higher clock frequency.
template <typename R, typename L>
void measure(const char *options, std::vector<float> &out, R &&runtime,
             L &&lut) {
  std::vector<float> reference;
  double runtime_ms = 1e300;
  double lut_ms = 1e300;
  for (int r = 0; r < runs; ++r) {
    runtime_ms = std::min(runtime_ms, milliseconds(runtime));
    if (r == 0) {
      reference = out;
    }
    lut_ms = std::min(lut_ms, milliseconds(lut));
  }
  report(options, runtime_ms, lut_ms, same(reference, out));
  total_runtime_ms += runtime_ms;
  total_lut_ms += lut_ms;
}

void header(const char *title) {
  std::printf("\n%s\n  %-34s %12s %12s %8s\n", title, "options", "run-time",
              "SpeciaLUT", "speedup");
  total_runtime_ms = total_lut_ms = 0;
}

// =============================================================================
// Case 1: trivial loop, one option
// =============================================================================

enum class Norm { squared, absolute, root, count };

template <typename Options>
ALWAYS_INLINE void norms(Options o, const float *__restrict in,
                         float *__restrict out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    switch (o.norm) {
    case Norm::squared:
      out[i] = in[i] * in[i];
      break;
    case Norm::absolute:
      out[i] = std::abs(in[i]);
      break;
    default:
      out[i] = std::sqrt(std::abs(in[i]));
      break;
    }
  }
}

struct NormAtRuntime {
  Norm norm;
};
template <Norm N> struct NormAtCompileTime {
  static constexpr Norm norm = N;
};

NOINLINE void norms_runtime(NormAtRuntime o, const float *in, float *out,
                            std::size_t n) {
  norms(o, in, out, n);
}

template <Norm N>
NOINLINE void norms_special(const float *in, float *out, std::size_t n) {
  norms(NormAtCompileTime<N>{}, in, out, n);
}

void benchmark_trivial() {
  constexpr std::size_t n = 1 << 14; // fits in the L1/L2 cache
  constexpr int sweeps = 4000;
  std::vector<float> in(n), out(n);
  for (std::size_t i = 0; i < n; ++i) {
    in[i] = float(i % 1000) / 100.0f - 5.0f;
  }

  const SpeciaLUT::Chooser<TABULATE(norms_special), Norm::count> chooser;
  const char *names[] = {"norm = squared", "norm = absolute", "norm = root"};

  header("1. Trivial loop, one option (the compiler does this on its own)");
  for (int k = 0; k < 3; ++k) {
    const Norm norm = opaque(static_cast<Norm>(k));
    measure(
        names[k], out,
        [&] {
          for (int s = 0; s < sweeps; ++s) {
            norms_runtime({norm}, in.data(), out.data(), n);
          }
        },
        [&] {
          const auto specialized = chooser(norm); // chosen once
          for (int s = 0; s < sweeps; ++s) {
            specialized(in.data(), out.data(), n);
          }
        });
  }
}

// =============================================================================
// Case 2: stencil with four orthogonal options
// =============================================================================

enum class Kernel { poly, spiky, rational, count };

struct Particles {
  std::vector<float> x, y, z, mass;
};

/// Sum of kernel weights over the neighbours of every particle, a density
/// estimate. Neighbours are the particles up to `reach` places away.
template <typename Options>
ALWAYS_INLINE void density(Options o, const float *__restrict x,
                           const float *__restrict y, const float *__restrict z,
                           const float *__restrict mass, float *__restrict out,
                           std::size_t n, float h) {
  constexpr std::ptrdiff_t reach = 8;
  const float inv_h2 = 1.0f / (h * h);

  for (std::size_t i = 0; i < n; ++i) {
    out[i] = 0.0f;
  }

  for (std::ptrdiff_t k = -reach; k <= reach; ++k) {
    const auto begin = static_cast<std::size_t>(reach);
    const std::size_t end = n - static_cast<std::size_t>(reach);
    for (std::size_t i = begin; i < end; ++i) {
      const std::size_t j = i + static_cast<std::size_t>(k);

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
        const float t = std::max(0.0f, 1.0f - q2);
        w = t * t * t;
      } break;
      case Kernel::spiky: {
        const float t = std::max(0.0f, 1.0f - std::sqrt(q2));
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
      out[i] += w;
    }
  }

  if (o.limited) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = std::min(out[i], 4.0f);
    }
  }
}

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

NOINLINE void density_runtime(DensityAtRuntime o, Particles const &p,
                              float *out, float h) {
  density(o, p.x.data(), p.y.data(), p.z.data(), p.mass.data(), out, p.x.size(),
          h);
}

template <Kernel K, int Dim, bool Weighted, bool Limited>
NOINLINE void density_special(Particles const &p, float *out, float h) {
  density(DensityAtCompileTime<K, Dim, Weighted, Limited>{}, p.x.data(),
          p.y.data(), p.z.data(), p.mass.data(), out, p.x.size(), h);
}

void benchmark_stencil() {
  constexpr std::size_t n = 1 << 14;
  constexpr int sweeps = 200;
  Particles p;
  for (std::size_t i = 0; i < n; ++i) {
    const float t = float(i);
    p.x.push_back(0.05f * t);
    p.y.push_back(0.2f * std::sin(0.37f * t));
    p.z.push_back(0.2f * std::cos(0.11f * t));
    p.mass.push_back(1.0f + 0.25f * std::sin(0.05f * t));
  }
  std::vector<float> out(n);
  const float h = 0.5f;

  // 3 kernels x 2 dimensions x 2 x 2 flags = 24 specializations
  const SpeciaLUT::Chooser<TABULATE(density_special), Kernel::count,
                           SpeciaLUT::values<2, 3>, 2, 2>
      chooser;

  header("2. Stencil with four orthogonal options (24 specializations)");
  const char *kernels[] = {"poly", "spiky", "rational"};
  for (int k = 0; k < 3; ++k) {
    for (int dim = 2; dim <= 3; ++dim) {
      for (int flags = 0; flags < 4; flags += 3) { // both off, both on
        const DensityAtRuntime o{opaque(static_cast<Kernel>(k)), opaque(dim),
                                 opaque(flags != 0), opaque(flags != 0)};
        char options[64];
        std::snprintf(options, sizeof(options), "%-8s %dD %s", kernels[k], dim,
                      flags ? "weighted, limited" : "plain");
        measure(
            options, out,
            [&] {
              for (int s = 0; s < sweeps; ++s) {
                density_runtime(o, p, out.data(), h);
              }
            },
            [&] {
              const auto specialized =
                  chooser(o.kernel, o.dim, o.weighted, o.limited);
              for (int s = 0; s < sweeps; ++s) {
                specialized(p, out.data(), h);
              }
            });
      }
    }
  }
  report("all of the above", total_runtime_ms, total_lut_ms, true);
}

// =============================================================================
// Case 3: choosing per element (how not to use it)
// =============================================================================

template <typename Options> ALWAYS_INLINE float one_norm(Options o, float v) {
  switch (o.norm) {
  case Norm::squared:
    return v * v;
  case Norm::absolute:
    return std::abs(v);
  default:
    return std::sqrt(std::abs(v));
  }
}

template <Norm N> float one_norm_special(float v) {
  return one_norm(NormAtCompileTime<N>{}, v);
}

NOINLINE void per_element_runtime(Norm norm, const float *__restrict in,
                                  float *__restrict out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = one_norm(NormAtRuntime{norm}, in[i]);
  }
}

NOINLINE void per_element_lut(Norm norm, const float *__restrict in,
                              float *__restrict out, std::size_t n) {
  const SpeciaLUT::Chooser<TABULATE(one_norm_special), Norm::count> chooser;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = chooser(norm)(in[i]); // a call through a pointer, per element
  }
}

void benchmark_per_element() {
  constexpr std::size_t n = 1 << 14;
  constexpr int sweeps = 4000;
  std::vector<float> in(n), out(n);
  for (std::size_t i = 0; i < n; ++i) {
    in[i] = float(i % 1000) / 100.0f - 5.0f;
  }

  header("3. Chosen for every element (how not to use it)");
  const Norm norm = opaque(Norm::squared);
  measure(
      "norm = squared", out,
      [&] {
        for (int s = 0; s < sweeps; ++s) {
          per_element_runtime(norm, in.data(), out.data(), n);
        }
      },
      [&] {
        for (int s = 0; s < sweeps; ++s) {
          per_element_lut(norm, in.data(), out.data(), n);
        }
      });
}

} // namespace

int main() {
  std::printf("SpeciaLUT benchmarks (best of %d runs each)\n", runs);

  benchmark_trivial();
  benchmark_stencil();
  benchmark_per_element();

  std::printf("\nspeedup = run-time branching / SpeciaLUT; above 1 the "
              "specialization is faster.\n");
  if (mismatches != 0) {
    std::printf("%d case(s) computed different results!\n", mismatches);
    return 1;
  }
  return 0;
}
