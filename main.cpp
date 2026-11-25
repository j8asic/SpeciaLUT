#include "specialut.hpp"
#include <functional>
#include <iostream>

// A function with both compile-time and run-time parameters
template<int i, int j, int k, int l>
void host_fn(const char* message)
{
    if constexpr (!l) {
        std::cout << "Compile-time specialization: (" << i << ", " << j << ", " << k << ", " << l
                  << "), and for this state I reject to print the message. " << std::endl;
    } else {
        std::cout << "Compile-time specialization: (" << i << ", " << j << ", " << k << ", " << l
                  << "), run-time message: " << message << std::endl;
    }
}

// Template parameters do not have to be plain indices: an enum, and a dimension that is either 2 or 3
enum class Kernel { Cubic, Quintic, Wendland, Count };

template<Kernel kernel, int dim, bool verbose>
void typed_fn(double radius)
{
    std::cout << "Kernel #" << int(kernel) << " in " << dim << "D";
    if constexpr (verbose)
        std::cout << ", with support radius " << radius;
    std::cout << std::endl;
}

// A function with the same template parameters as typed_fn, which has nothing to offer for one of their combinations
template<Kernel kernel, int dim, bool verbose>
    requires(!(kernel == Kernel::Cubic && dim == 3))
void picky_fn(double radius)
{
    std::cout << "Picky about kernel #" << int(kernel) << " in " << dim << "D";
    if constexpr (verbose)
        std::cout << ", with support radius " << radius;
    std::cout << std::endl;
}

// A function specialized only for the common sizes
template<int width>
int sum_fixed(const int* data)
{
    int sum = 0;
    for (int i = 0; i < width; ++i) // the trip count is a constant here
        sum += data[i];
    return sum;
}

int sum_generic(const int* data, int width)
{
    int sum = 0;
    for (int i = 0; i < width; ++i)
        sum += data[i];
    return sum;
}

struct SomeStruct
{
    int member{};

    // A member function with both compile-time and run-time parameters
    template<int i>
    void member_fn(const char* message)
    {
        std::cout << "Compile-time specialization: (" << i << "), run-time message: " << message
                  << ", member value: " << member << std::endl;
    }

    void test_from_within()
    {
        constexpr auto mfn = TABULATE(SomeStruct::member_fn);
        SpeciaLUT::Chooser<mfn, 3> chooser;
        (this->*chooser(2))("Hello from an object, using ugly pointer calls");
        chooser(this, 2)("Hello from an object, using our nicer syntax");
    }
};

#ifdef USE_CUDA
// A CUDA kernel with both compile-time and run-time parameters
template<bool i, bool j, int k, bool l>
__global__ void cuda_fn(int msg, float scale)
{
    printf("CUDA kernel specialization: (%i, %i, %i, %i), run-time number: %i, scaled: %.1f\n", i, j, k, l, msg,
           scale * float(msg));
}
#endif

struct Functor
{
    template<bool yes>
    void operator()(int num)
    {
        if constexpr (yes)
            std::cout << "Yes " << num << std::endl;
        else
            std::cout << "No " << num << std::endl;
    }
};

int main()
{
    // prepare runtime parameters
    int o = 0, l = 1, z = 2;

    // build a table of host_fn specializations, for which parameters have finite number of states
    std::cout << "Simple test:" << std::endl;
    SpeciaLUT::Chooser<TABULATE(host_fn), 2, 2, 3, 2> test;
    // choose the specialization and run it with a runtime parameter
    test(o, o, z, l)("Hello #1");
    test(o, l, o, o)("Hey #2");
    test(l, o, l, l)("Hi #3");
    test(l, l, z, o)("Aloha #4");

    // every parameter is checked against its own number of states
    try {
        test(z, o, o, o)("never printed");
    } catch (const std::out_of_range& e) {
        std::cout << "State (2, 0, 0, 0) is refused: " << e.what() << std::endl;
    }

    // parameters described by what they are, instead of by a number of states:
    //  - Kernel::Count  the enumerators before Count, chosen by a Kernel
    //  - values<2, 3>   exactly these values, chosen by value
    //  - 2              a bool
    std::cout << "\nNow testing typed parameters:" << std::endl;
    SpeciaLUT::Chooser<TABULATE(typed_fn), Kernel::Count, SpeciaLUT::values<2, 3>, 2> typed;
    Kernel kernel = Kernel::Wendland;
    int dim = 3;
    typed(kernel, dim, true)(2.5);
    typed(Kernel::Cubic, 2, false)(2.5);

    // functions that are specialized by the same parameters: describe the parameters once,
    // choose (and check) their states once, and use them with the table of every function
    std::cout << "\nNow testing states shared by two tables:" << std::endl;
    using Options = SpeciaLUT::States<Kernel::Count, SpeciaLUT::values<2, 3>, 2>;
    Options::Chooser<TABULATE(typed_fn)> typed_table;
    Options::Chooser<TABULATE(picky_fn)> picky_table;
    const Options options(kernel, dim, false);
    typed_table[options](2.5);
    picky_table[options](2.5);

    // picky_fn rejects a cubic kernel in 3D by its requires-clause: these specializations
    // are not compiled, and the table has nothing at their place
    std::cout << "Picky table: " << picky_table.size() << " specializations for " << Options::size()
              << " combinations of states" << std::endl;
    if (!picky_table.find(Kernel::Cubic, 3, true))
        std::cout << "Nothing for a cubic kernel in 3D" << std::endl;

    // find() returns nullptr for the states that are not in the table,
    // so the common cases can be specialized and the rest left to generic code
    std::cout << "\nNow testing the fallback to a generic function:" << std::endl;
    SpeciaLUT::Chooser<TABULATE(sum_fixed), SpeciaLUT::values<4, 8>> sums;
    const int data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    for (int width : { 4, 6, 8 }) {
        if (auto specialized = sums.find(width))
            std::cout << "Sum of " << width << " (specialized): " << specialized(data) << std::endl;
        else
            std::cout << "Sum of " << width << " (generic): " << sum_generic(data, width) << std::endl;
    }

    // build a table of SomeStruct::member_fn specializations, for which parameters have finite number of states
    std::cout << "\nNow testing the member-function specialization:" << std::endl;
    SpeciaLUT::Chooser<TABULATE(SomeStruct::member_fn), 3> member_chooser;
    // create an instance onto which the chosen specialization will be run
    SomeStruct object{ 1333 };
    // choose the specialization and run it, using any of the below ugly syntax
    (object.*member_chooser(o))("Ugly (class.*member)(arguments) call");
    std::invoke(member_chooser(l), object, "Ugly std::invoke call");
    member_chooser(object, z)("Our nicer syntax");
    // call the thing from a member function
    object.test_from_within();

    // example using a lambda expression (note the different macro)
    std::cout << "\nNow testing the lambda specialization:" << std::endl;
    auto lam = [&]<bool T>(int x) -> void { std::cout << "Lambda outputs: " << int(T) + x * z << std::endl; };
    SpeciaLUT::Chooser<TABULATE_LAMBDA(lam), 2> lambdas;
    // lambdas are nothing but structs with operator()
    (lam.*lambdas(true))(o);
    lambdas(lam, false)(l);
    // or with no macro and no table object at all
    SpeciaLUT::choose_lambda<2>(lam, true)(z);
    // which also is the shortest way to dispatch to any template, from where it is needed
    SpeciaLUT::choose_lambda<Kernel::Count, SpeciaLUT::values<2, 3>>(
      [&]<Kernel K, int D>(double radius) { typed_fn<K, D, true>(radius); }, kernel, dim)(1.5);

    // example using a functor object
    std::cout << "\nNow testing the functor object:" << std::endl;
    SpeciaLUT::Chooser<TABULATE_FUNCTOR(Functor), 2> functor_table;
    Functor fun;
    (fun.*functor_table(true))(l);
    functor_table(fun, false)(o);

#ifdef USE_CUDA

    std::cout << "\nNow testing the CUDA kernel:" << std::endl;

    // build a table of cuda_fn specializations, for which parameters have finite number of states
    SpeciaLUT::CudaChooser<TABULATE(cuda_fn), 2, 2, 3, 2> cuda_test;

    // set grid size and block size for kernel execution
    cuda_test.prepare({ 1, 1, 1 }, { 1, 1, 1 });

    // choose the specialization and immediatelly launch the kernel with runtime parameters,
    // which are converted to the parameter types of the kernel (here: 0.5 to float)
    bool launched = cuda_test(o, o, z, l)(1, 0.5) == cudaSuccess;
    launched &= cuda_test(o, l, o, o)(2, 0.5) == cudaSuccess;
    launched &= cuda_test(l, o, l, l)(3, 0.5) == cudaSuccess;
    // cuda_test(...) returns a CudaKernel, so you can use cuda_test(...).prepare(...).launch(...)
    // to change execution parameters for that one launch
    launched &= cuda_test(l, l, z, o).prepare({ 1, 1, 1 }, { 2, 1, 1 }).launch(4, 0.5) == cudaSuccess;

    if (!launched || cudaDeviceSynchronize() != cudaSuccess) {
        std::cout << "CUDA kernels failed: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return 1;
    }
#endif
}
