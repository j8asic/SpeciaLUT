# SpeciaLUT
Run-time choosing of template specializations using compile-time lookup-tables (LUT). Simply put: compile all possible states of a template function, but execute the optimal one at run-time.

## Use cases

Heavy functions can have many branching scenarios inside loops. Some branches may be dependent on conditions that do not change during execution of the function. Branching prediction works good on CPUs, but in any case, conditions and jumping takes time. Moreover, GPUs do not use branching prediction. 

Therefore, the safest thing is that the non-used parts of the code are not even there -- that the compiler deletes them. This is possible by introducing compile-time conditions, if the conditions are immutable during the function execution.

## How it works

1. User defines number of states that each condition (template parameter) has.
2. Compiler compiles all possible specializations of the function, and stores the function pointers in a lookup table (LUT).
3. When user wants to execute the function with run-time parameters, the optimal one is executed.

## How to use it

**Requirements**: C++17 compiler (enabled with `-std=c++17`)

**Test**: Run CMake as usual, or open the project in an IDE.

**Set up**: Copy `specialut.hpp` into your project and include it.

**Example**:

Include the library:

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
using table = TABULATE(run);
SpeciaLUT::Chooser<table, 2, 3> test;
```

Find the optimal function based on the run-time conditions (first brackets), pass other parameters and execute the function (second brackets).

```cpp
test(runtime_bool, int_state)(double_parameter);
```

There is a construct for CUDA kernels as well (see `main.cpp` for an example), which is used as:

```cpp
using table = TABULATE(some_cuda_kernel);
SpeciaLUT::CudaChooser<table, 2, 3> test;
```

[Try online in Compiler Explorer](https://godbolt.org/z/hMq4KzYET)


## Be aware of ...

slow compilation of large functions. This thing compiles all possible specializations. E.g. if you have 3 parameters with 3 states, it will compile 3^3 = 27 functions.

## Tested on

- Clang >= 5.0
- GCC >= 8.1

## Roadmap

- C++ non-member functions (DONE)
- C++ member functions (DONE)
- CUDA kernels (DONE)
- HIP kernels (maybe)
- C++17 workarounds (DONE)

## License

BSD 2-Clause License
Copyright (c) 2022, Josip Basic
