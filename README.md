# SpeciaLUT

Run-time choosing of template specializations using compile-time lookup-tables (LUT). Simply put: compile all possible states of a template function, but execute the optimal one at run-time.

## Ideal Use Case

SpeciaLUT is designed for **hot functions** with:
- **Many iterations inside** the function (loops over large data)
- **Multiple if/else or enum-based control flow** that creates branching
- **Conditions that are loop-invariant** (set once before the function, used many times inside)

The library pre-compiles all possible specializations and selects the right one at runtime via a lookup table. Inside a specialization the conditions are constants, so the compiler removes the branches and the code behind them, and optimizes what is left: it vectorizes the loop, unrolls loops whose length became a constant, keeps fewer values alive.

**Where it is applicable:**
- Large functions that can't be inlined
- Code with many orthogonal boolean/enum flags, or with a few common sizes (2D/3D, block of 4/8/16)
- GPU kernels (CUDA/HIP) — smaller kernels without branch instructions, dead code and the registers it needs

**Where it is not applicable:**
- Small functions that the compiler can inline and optimize
- Conditions that vary per-iteration (use runtime branching instead)

See [Does it pay off?](#does-it-pay-off) for measurements of both.

## How it works

1. User defines number of states that each condition (template parameter) has.
2. Compiler compiles all possible specializations of the function, and stores the function pointers in a lookup table (LUT).
3. When user wants to execute the function with run-time parameters, the optimal one is executed.

## How to use it

**Requirements**: C++20 compiler (enabled with `-std=c++20`)

**Set up**: Copy `specialut.hpp` into your project and include it. With CMake, `add_subdirectory` this repository and link to `SpeciaLUT::SpeciaLUT` instead.

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

The first brackets return a plain function pointer, so in a loop you can choose once and keep it.

[Try online in Compiler Explorer](https://godbolt.org/z/Gzc87hPG6)

### Describing the template parameters

Each template parameter of the function is described by one argument of `Chooser`, in the same order:

| You write | The template is instantiated with | At run-time it is chosen by |
|-----------|-----------------------------------|-----------------------------|
| `N` | `0` .. `N-1` (write `2` for a `bool`) | anything convertible to an index |
| `Enum::Count` | the enumerators before `Count`, as `Enum` | an `Enum` |
| `SpeciaLUT::values<A, B, ...>` | exactly the listed values | a value, not its position |

```cpp
enum class Kernel { Cubic, Quintic, Wendland, Count };

template<Kernel kernel, int dim, bool viscous>
void interact(Particles& particles);

// 3 kernels x 2 dimensions x 2 = 12 specializations
SpeciaLUT::Chooser<TABULATE(interact), Kernel::Count, SpeciaLUT::values<2, 3>, 2> chooser;

chooser(Kernel::Wendland, 3, true)(particles);
```

The number of run-time parameters and their types are checked at compile-time: here the first one has to be a `Kernel`, a bare number is refused.

### States that are not in the table

Every run-time parameter is checked against its own range. A parameter outside of it throws `std::out_of_range` (or calls `std::abort()` if exceptions are disabled); it never runs some other specialization.

`find` returns `nullptr` instead. Use it to specialize the common cases only, and leave the rest to generic code:

```cpp
template<int width> void filter_fixed(Image& image);  // loops unrolled, vectorized
void filter_generic(Image& image, int width);

SpeciaLUT::Chooser<TABULATE(filter_fixed), SpeciaLUT::values<3, 5, 7>> filters;

if (auto specialized = filters.find(width))
    specialized(image);
else
    filter_generic(image, width);
```

`chooser(...)`, `Chooser<...>::at(...)` and `Chooser<...>::find(...)` are `constexpr`, and the latter two are static, so no object is needed for them. With C++23 the call operator is static too.

### Combinations that make no sense

If the template rejects a combination of its parameters with a `requires`-clause, that combination is not compiled, and its place in the table stays empty. `find` returns `nullptr` for it, and choosing it in any other way is refused like a state outside of its range.

```cpp
template<Kernel kernel, int dim, bool viscous>
    requires(!(kernel == Kernel::Cubic && viscous))  // there is no viscous cubic kernel
void interact(Particles& particles);

using Interact = SpeciaLUT::Chooser<TABULATE(interact), Kernel::Count, SpeciaLUT::values<2, 3>, 2>;

static_assert(Interact::size() == 10);  // specializations, for 12 combinations of states
```

`size()` is the number of specializations that were compiled, so a `static_assert` like the one above keeps it from changing unnoticed.

### Several functions with the same parameters

A solver has many functions specialized by the same options. Describe the options once with `States`, choose them once, and use the result with the table of every function:

```cpp
using Options = SpeciaLUT::States<Kernel::Count, SpeciaLUT::values<2, 3>, 2>;

Options::Chooser<TABULATE(density)> density;
Options::Chooser<TABULATE(forces)> forces;
Options::Chooser<TABULATE(integrate)> integrate;

const Options options(kernel, dim, viscous);  // checked here, once

density[options](particles);
forces[options](particles);
integrate[options](particles, time_step);
```

`options` holds one index, so `density[options]` is a single load from the table with nothing left to check, and all the functions are sure to run the same variant. `Options::contains(...)` tells if the run-time parameters are valid without throwing. For member functions there is `chooser(object, options)(arguments)`, and for kernels `Options::CudaChooser<...>`.

### Member functions, functors and lambdas

```cpp
SpeciaLUT::Chooser<TABULATE(SomeStruct::member_fn), 3> chooser;

chooser(object, state)(arguments);       // bound to an object (or a pointer to it)
(object.*chooser(state))(arguments);     // the same, with the bare member pointer

SpeciaLUT::Chooser<TABULATE_FUNCTOR(Functor), 2> functors;
SpeciaLUT::Chooser<TABULATE_LAMBDA(lambda), 2> lambdas;
```

`choose_lambda` needs no macro and no table object. It is also the shortest way to dispatch to any template right where it is needed:

```cpp
SpeciaLUT::choose_lambda<Kernel::Count, 2>(
    [&]<Kernel K, bool V>() { interact<K, 3, V>(particles); }, kernel, viscous)();
```

### CUDA and HIP kernels

```cpp
SpeciaLUT::CudaChooser<TABULATE(some_kernel), 2, 3> kernel;
kernel.prepare(grid_dim, block_dim);
kernel(runtime_bool, int_state)(kernel_args);
```

The last brackets launch the kernel and return the error code of the runtime (`cudaError_t` or `hipError_t`). The arguments are converted to the parameter types of the kernel, like in an ordinary call, and a wrong number of them does not compile. `kernel(...)` returns the chosen kernel, so `kernel(...).prepare(...).launch(...)` changes the execution parameters for a single launch. `kernel[options]` chooses by `States`, as above.

The GPU part of the header is enabled when the file is compiled as CUDA or HIP code.

### Configuration macros

| Macro | Effect |
|-------|--------|
| `SPECIALUT_NO_SHORT_MACROS` | Do not define `TABULATE`, `TABULATE_FUNCTOR` and `TABULATE_LAMBDA`; use the same names prefixed with `SPECIALUT_` |
| `SPECIALUT_NO_GPU` | Leave the GPU part out, even in CUDA or HIP code |
| `SPECIALUT_USE_CUDA`, `SPECIALUT_USE_HIP` | Enable the GPU part for that runtime, whatever the compiler says |
| `SPECIALUT_NO_CUDA_ALIASES` | With HIP: do not define `cudaStream_t`, `cudaError_t`, `cudaSuccess` and `cudaLaunchKernel` as aliases of their HIP counterparts |

## Does it pay off?

`benchmark.cpp` compares a specialization chosen through SpeciaLUT with the same source code taking its options at run-time. The numbers below are run-time branching divided by SpeciaLUT, so above 1 the specialization is faster. They were measured on an AMD Ryzen 9 5950X with `-O3 -march=native`.

| Case | GCC 16 | Clang 23 |
|------|--------|----------|
| Trivial loop with one option | 1.0x | 1.0x |
| Stencil with four options (24 specializations), all measured combinations | 2.0x | 1.7x |
| – combinations where the specialization gets vectorized | 4.8x – 6.5x | 5.6x – 6.5x |
| – kernel with `std::sqrt`, which keeps both variants scalar | 1.0x – 1.1x | 1.0x |
| – combinations which the compiler specializes on its own | none | 1.0x (2 of 12) |
| The same stencil with `-fno-math-errno` | 5.6x | 1.1x |
| Tiny function chosen for every element | 0.06x | 1.0x |

What to take from it:

- A branch on a loop-invariant condition is predicted well and costs almost nothing by itself. The gain comes from **what the compiler does once the condition is a constant**, here vectorizing the loop. Where it has nothing to gain (the kernel with `std::sqrt`, which may set `errno` and so is not vectorized), the specialization is not faster.
- Compilers can hoist such branches out of a loop on their own (loop unswitching), which is why the trivial loop gains nothing. Whether they do it for a real loop depends on the compiler, its flags and details of the loop body: with `-fno-math-errno` Clang does almost all of it for this stencil and GCC none of it. SpeciaLUT makes it happen regardless.
- Choose once, outside of the hot loop. A call through a function pointer per element is 16 times slower with GCC here, unless the compiler manages to see through the table, as Clang does.

`benchmark_cuda.cu` runs the same stencil as a CUDA kernel. The options are the same for every thread, so the kernel that branches at run-time has no warp divergence to pay for; the specialization can only win what the compiler makes of the constants. Expect less than on the CPU, and measure your own kernels.

## Files

| File | Description |
|------|-------------|
| `specialut.hpp` | Header-only library — copy this into your project |
| `main.cpp` | Usage examples: free functions, typed parameters, member functions, lambdas, functors, CUDA |
| `benchmark.cpp` | Performance comparison: LUT dispatch vs runtime branching |
| `benchmark_cuda.cu` | The same comparison for CUDA kernels |
| `tests/` | Tests, for the host and for CUDA kernels |

## Building the examples and the tests

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
build/SpeciaLUT_benchmark
```

The CUDA example, tests and benchmark are built if CMake finds a CUDA compiler; `-DSPECIALUT_CUDA=OFF` leaves them out. Other options: `SPECIALUT_BUILD_EXAMPLES`, `SPECIALUT_BUILD_TESTS`, `SPECIALUT_NATIVE` (tune the benchmark for the CPU it is built on), `SPECIALUT_SANITIZE` and `SPECIALUT_WARNINGS_AS_ERRORS`.

## Be aware of ...

- Slow compilation of large functions. This compiles all possible specializations. E.g. if you have 4 boolean parameters, it will compile 2^4 = 16 functions. Tabulate only the values you need with `SpeciaLUT::values`, leave the rare ones to generic code through `find`, and rule out the combinations that make no sense with a `requires`-clause.
- A combination that the template cannot be instantiated with is left out of the table, whatever the reason. A `bool` parameter tabulated with 3 states gives a table of 2 specializations, not a compile error. Only a function that cannot be specialized for any combination is an error.
- All specializations must have the same signature (the same run-time parameters and return type), as they share one function pointer type.
- Results can differ in the last bits between specializations and a run-time version, when the compiler contracts floating-point operations (FMA) differently in differently optimized code. `-ffp-contract=off` makes them identical.

## Tested on

This version:
- GCC 13.4, 15.3 and 16.2
- Clang 20.1 and 23.1
- nvcc 13.5 (CUDA), kernels run on RTX 2080 Ti and RTX 4090

each with `-std=c++20` and `-std=c++23`.

Versions before the typed parameters (`Enum::Count`, `values<...>`) were added, not verified again since:
- Clang >= 13.0
- Clang 12 requires that `auto table = TABULATE(run)` is done, and `table` passed to `Chooser`
- GCC >= 10.1
- GCC 9.4 requires `-std=c++2a` instead of `-std=c++20`
- MSVC >= 19.30 requires that `constexpr auto table = TABULATE(run)` is done, and `table` passed to `Chooser`
- HIP kernels

Clang can compile the CUDA parts too (`-x cuda`), but needs a CUDA toolkit it supports to run them.

## Roadmap

- Non-member functions (DONE)
- Member functions (DONE)
- Lambdas and functors (DONE)
- CUDA kernels (DONE)
- HIP kernels (DONE)
- C++20, C++23 features (DONE)
- Enum parameters, explicit lists of values, fallback for states outside of the table (DONE)
- One choice of states shared by the tables of many functions (DONE)
- Combinations rejected by a `requires`-clause are not compiled (DONE)
- Tests (DONE)
- C++17 workarounds (DONE, but not maintained, checkout branch `cxx17`)

## License

BSD 2-Clause License
Copyright (c) 2022, Josip Basic

