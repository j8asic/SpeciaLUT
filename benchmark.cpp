// SpeciaLUT Benchmarks
// Demonstrates scenarios where LUT-based dispatch outperforms runtime branching

#include "specialut.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Prevent inlining to simulate real-world large functions
#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE
#endif

// =============================================================================
// Benchmark 1: Single mode parameter with heavy computation
// Shows benefit when function is too large to inline
// =============================================================================

template <int Mode>
NOINLINE float compute_single(const float *__restrict data, size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    if constexpr (Mode == 0) {
      sum += data[i] * data[i];
    } else if constexpr (Mode == 1) {
      sum += std::sqrt(std::abs(data[i]));
    } else {
      sum += std::log1p(std::abs(data[i]));
    }
  }
  return sum;
}

NOINLINE float compute_single_runtime(const float *data, size_t n, int mode) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    if (mode == 0) {
      sum += data[i] * data[i];
    } else if (mode == 1) {
      sum += std::sqrt(std::abs(data[i]));
    } else {
      sum += std::log1p(std::abs(data[i]));
    }
  }
  return sum;
}

// =============================================================================
// Benchmark 2: Multiple orthogonal conditions (combinatorial explosion)
// With 4 boolean flags = 16 runtime branches vs 16 specialized functions
// =============================================================================

template <bool UseAbs, bool UseScale, bool UseBias, bool UseClamp>
NOINLINE float compute_multi(const float *__restrict data, size_t n,
                             float scale, float bias) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    float v = data[i];
    // Nested if constexpr mirrors the runtime nested if/else structure
    if constexpr (UseAbs) {
      if constexpr (UseScale) {
        if constexpr (UseBias) {
          if constexpr (UseClamp) {
            v = std::max(0.0f, std::min(1.0f, std::abs(v) * scale + bias));
          } else {
            v = std::abs(v) * scale + bias;
          }
        } else {
          if constexpr (UseClamp) {
            v = std::max(0.0f, std::min(1.0f, std::abs(v) * scale));
          } else {
            v = std::abs(v) * scale;
          }
        }
      } else {
        if constexpr (UseBias) {
          if constexpr (UseClamp) {
            v = std::max(0.0f, std::min(1.0f, std::abs(v) + bias));
          } else {
            v = std::abs(v) + bias;
          }
        } else {
          if constexpr (UseClamp) {
            v = std::max(0.0f, std::min(1.0f, std::abs(v)));
          } else {
            v = std::abs(v);
          }
        }
      }
    } else {
      if constexpr (UseScale) {
        if constexpr (UseBias) {
          if constexpr (UseClamp) {
            v = std::max(0.0f, std::min(1.0f, v * scale + bias));
          } else {
            v = v * scale + bias;
          }
        } else {
          if constexpr (UseClamp) {
            v = std::max(0.0f, std::min(1.0f, v * scale));
          } else {
            v = v * scale;
          }
        }
      } else {
        if constexpr (UseBias) {
          if constexpr (UseClamp) {
            v = std::max(0.0f, std::min(1.0f, v + bias));
          } else {
            v = v + bias;
          }
        } else {
          if constexpr (UseClamp) {
            v = std::max(0.0f, std::min(1.0f, v));
          } else {
            // v = v; (no-op)
          }
        }
      }
    }
    sum += v;
  }
  return sum;
}

NOINLINE float compute_multi_runtime(const float *data, size_t n, float scale,
                                     float bias, bool use_abs, bool use_scale,
                                     bool use_bias, bool use_clamp) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    float v = data[i];
    // Real-world code typically uses nested if/else, not linear ifs
    if (use_abs) {
      if (use_scale) {
        if (use_bias) {
          if (use_clamp) {
            v = std::max(0.0f, std::min(1.0f, std::abs(v) * scale + bias));
          } else {
            v = std::abs(v) * scale + bias;
          }
        } else {
          if (use_clamp) {
            v = std::max(0.0f, std::min(1.0f, std::abs(v) * scale));
          } else {
            v = std::abs(v) * scale;
          }
        }
      } else {
        if (use_bias) {
          if (use_clamp) {
            v = std::max(0.0f, std::min(1.0f, std::abs(v) + bias));
          } else {
            v = std::abs(v) + bias;
          }
        } else {
          if (use_clamp) {
            v = std::max(0.0f, std::min(1.0f, std::abs(v)));
          } else {
            v = std::abs(v);
          }
        }
      }
    } else {
      if (use_scale) {
        if (use_bias) {
          if (use_clamp) {
            v = std::max(0.0f, std::min(1.0f, v * scale + bias));
          } else {
            v = v * scale + bias;
          }
        } else {
          if (use_clamp) {
            v = std::max(0.0f, std::min(1.0f, v * scale));
          } else {
            v = v * scale;
          }
        }
      } else {
        if (use_bias) {
          if (use_clamp) {
            v = std::max(0.0f, std::min(1.0f, v + bias));
          } else {
            v = v + bias;
          }
        } else {
          if (use_clamp) {
            v = std::max(0.0f, std::min(1.0f, v));
          } else {
            // v = v; (no-op)
          }
        }
      }
    }
    sum += v;
  }
  return sum;
}


// =============================================================================
// Timer utility
// =============================================================================

class Timer {
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_;

public:
  Timer() : start_(Clock::now()) {}
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_)
        .count();
  }
};

// =============================================================================
// Main benchmark runner
// =============================================================================

int main() {
  constexpr size_t N = 5'000'000;
  constexpr int ITERATIONS = 50;

  std::vector<float> data(N);
  for (size_t i = 0; i < N; ++i) {
    data[i] = static_cast<float>(i % 1000) / 100.0f - 5.0f; // -5 to +5
  }

  volatile float sink = 0; // Prevent dead code elimination

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "SpeciaLUT Benchmarks\n";
  std::cout << "====================\n";
  std::cout << "Data size: " << N << " elements, " << ITERATIONS
            << " iterations\n\n";

  // -------------------------------------------------------------------------
  // Benchmark 1: Single mode parameter
  // -------------------------------------------------------------------------
  {
    std::cout << "Benchmark 1: Single mode (non-inlinable function)\n";
    std::cout << "-------------------------------------------------\n";

    SpeciaLUT::Chooser<TABULATE(compute_single), 3> chooser;
    volatile int mode = 1;

    // Warmup
    sink = compute_single_runtime(data.data(), N, mode);
    sink = chooser(mode)(data.data(), N);

    Timer t_runtime;
    for (int i = 0; i < ITERATIONS; ++i) {
      sink = compute_single_runtime(data.data(), N, mode);
    }
    double runtime_ms = t_runtime.elapsed_ms();

    Timer t_lut;
    for (int i = 0; i < ITERATIONS; ++i) {
      sink = chooser(mode)(data.data(), N);
    }
    double lut_ms = t_lut.elapsed_ms();

    std::cout << "  Runtime branching: " << runtime_ms << " ms\n";
    std::cout << "  LUT dispatch:      " << lut_ms << " ms\n";
    std::cout << "  Speedup:           " << runtime_ms / lut_ms << "x\n\n";
  }

  // -------------------------------------------------------------------------
  // Benchmark 2: Multiple orthogonal conditions (4 bools = 16 combinations)
  // -------------------------------------------------------------------------
  {
    std::cout << "Benchmark 2: 4 boolean flags (16 combinations)\n";
    std::cout << "-----------------------------------------------\n";

    SpeciaLUT::Chooser<TABULATE(compute_multi), 2, 2, 2, 2> chooser;
    volatile bool use_abs = true, use_scale = true;
    volatile bool use_bias = false, use_clamp = true;
    float scale = 2.0f, bias = 0.5f;

    // Warmup
    sink = compute_multi_runtime(data.data(), N, scale, bias, use_abs,
                                 use_scale, use_bias, use_clamp);
    sink = chooser(use_abs, use_scale, use_bias, use_clamp)(data.data(), N,
                                                            scale, bias);

    Timer t_runtime;
    for (int i = 0; i < ITERATIONS; ++i) {
      sink = compute_multi_runtime(data.data(), N, scale, bias, use_abs,
                                   use_scale, use_bias, use_clamp);
    }
    double runtime_ms = t_runtime.elapsed_ms();

    Timer t_lut;
    for (int i = 0; i < ITERATIONS; ++i) {
      sink = chooser(use_abs, use_scale, use_bias, use_clamp)(data.data(), N,
                                                              scale, bias);
    }
    double lut_ms = t_lut.elapsed_ms();

    std::cout << "  Runtime branching: " << runtime_ms << " ms\n";
    std::cout << "  LUT dispatch:      " << lut_ms << " ms\n";
    std::cout << "  Speedup:           " << runtime_ms / lut_ms << "x\n\n";
  }

  // -------------------------------------------------------------------------
  // Summary
  // -------------------------------------------------------------------------
  std::cout << "Key Takeaways:\n";
  std::cout << "--------------\n";
  std::cout << "SpeciaLUT is designed for HOT FUNCTIONS with:\n";
  std::cout << "  - Many iterations INSIDE the function\n";
  std::cout << "  - Multiple if/else or enum-based control flow\n";
  std::cout << "  - Conditions that are LOOP-INVARIANT (set once, used many times)\n";
  std::cout << "\n";
  std::cout << "Benefits are most pronounced on:\n";
  std::cout << "  - GPU kernels (avoids warp divergence)\n";
  std::cout << "  - Large functions that can't be inlined\n";
  std::cout << "  - Complex control flow that defeats cmov optimization\n";

  return 0;
}
