# SpeciaLUT
Run-time choosing of template specializations using compile-time lookup-tables (LUT). Simply put: compile all possible states of a templated functions, but execute the optimal one at run-time.

## Use cases

Heavy functions can have many branching scenarios inside loops. Some branches may be dependent on conditions that do not change during execution of the function. Branching prediction works good on CPUs, but in any case, conditions and jumping takes time. Moreover, GPUs do not use branching prediction. 

Therefore, the safest thing is that the non-used parts of the code are not even there -- that the compiler deletes them. This is possible if the conditions are immutable during the function execution.

## How it works

1. User defines number of states that each condition (template parameter) has.

2. Compiler compiles all possible specializations of the function, and stores the function pointers in a lookup table (LUT).
3. When user wants to execute the function with run-time parameters, the optimal one is executed.

## How to use it

**Requirements**: C++20 compiler.

**Test**: Run CMake or open the project in an IDE.

**Usage**: Copy `specialut.hpp` into your project and include it.

**Example**:

Include the library:

```cpp
#include "specialut.hpp"
```

Make a struct with a static function named `run`:

```cpp
struct Test
{
    template<bool condition, int state>
    static void run(double some_param) { /* ... */ }
};
```

Make an instance of `Chooser` class, with template parameters that specify: the *struct*, *function signature*, *number of states* for each template parameter:

```cpp
SpeciaLUT::Chooser<Test, void(double), 2, 3> test;
```

Find the optimal function based on the immutable run-time conditions (first brackets), pass other parameters and  execute the function (second brackets).

```cpp
test(runtime_bool, int_state)(double_parameter);
```

## TODO

- CUDA functions
- HIP functions