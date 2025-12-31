# SpeciaLUT

Run-time choosing of template specializations using compile-time lookup-tables (LUT). Simply put: compile all possible states of a template function, but execute the optimal one at run-time.

## Ideal Use Case

SpeciaLUT is designed for **hot functions** with:
- **Many iterations inside** the function (loops over large data)
- **Multiple if/else or enum-based control flow** that creates branching
- **Conditions that are loop-invariant** (set once before the function, used many times inside)

The library pre-compiles all possible specializations and selects the right one at runtime via a lookup table. This eliminates branching overhead inside tight loops.

**Where it is applicable:**
- GPU kernels (CUDA/HIP) — avoids warp divergence
- Large functions that can't be inlined
- Code with many orthogonal boolean/enum flags

**Where it is not applicable:**
- Small functions that the compiler can inline and optimize
- Conditions that vary per-iteration (use runtime branching instead)

## How it works

1. User defines number of states that each condition (template parameter) has.
2. Compiler compiles all possible specializations of the function, and stores the function pointers in a lookup table (LUT).
3. When user wants to execute the function with run-time parameters, the optimal one is executed.

## How to use it

**Requirements**: C++20 compiler (enabled with `-std=c++20`)

**Set up**: Copy `specialut.hpp` into your project and include it.

**Example**:

```cpp
#include "specialut.hpp"
```

A function `run` has both compile-time and run-time parameters:

```cpp
template<bool condition, int state>
void run(double some_param) {

    while (loop_condition) {

        if constexpr (condition) {
            /* ... */
        } else {
            /* ... */
        }

        if constexpr (state == SOME_ENUM) {
            /* ... */
        }
    }

}
```

Make an instance of `Chooser` class that requires: the template function and *number of states* for each template parameter:

```cpp
SpeciaLUT::Chooser<TABULATE(run), 2, 3> chooser;
```

Choose the specialization at runtime (first brackets), then call it with function arguments (second brackets):

```cpp
chooser(runtime_bool, int_state)(double_parameter);
```

For CUDA/HIP kernels:

```cpp
SpeciaLUT::CudaChooser<TABULATE(some_kernel), 2, 3> kernel;
kernel.prepare(grid_dim, block_dim);
kernel(runtime_bool, int_state)(kernel_args);
```

[Try online in Compiler Explorer](https://godbolt.org/z/Gzc87hPG6)

## Files

| File | Description |
|------|-------------|
| `specialut.hpp` | Header-only library — copy this into your project |
| `main.cpp` | Usage examples: free functions, member functions, lambdas, functors, CUDA |
| `benchmark.cpp` | Performance comparison: LUT dispatch vs runtime branching |

## Be aware of ...

Slow compilation of large functions. This compiles all possible specializations. E.g. if you have 4 boolean parameters, it will compile 2^4 = 16 functions.

## Tested on

- Clang >= 13.0
- Clang 12 requires that `auto table = TABULATE(run)` is done, and `table` passed to `Chooser`
- GCC >= 10.1
- GCC 9.4 requires `-std=c++2a` instead of `-std=c++20`
- MSVC >= 19.30 requires that `constexpr auto table = TABULATE(run)` is done, and `table` passed to `Chooser`

## Roadmap

- Non-member functions (DONE)
- Member functions (DONE)
- Lambdas and functors (DONE)
- CUDA kernels (DONE)
- HIP kernels (DONE)
- C++20, C++23 features (DONE)
- C++17 workarounds (DONE, but not maintained, checkout branch `cxx17`)

## License

BSD 2-Clause License
Copyright (c) 2022, Josip Basic

