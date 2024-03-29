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
__global__ void cuda_fn(int msg)
{
    int const thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("CUDA kernel specialization: (%i, %i, %i, %i), run-time number: %i\n", i, j, k, l, msg);
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

    // choose the specialization and immediatelly launch the kernel with a runtime parameter
    cuda_test(o, o, z, l)(1);
    cuda_test(o, l, o, o)(2);
    cuda_test(l, o, l, l)(3);
    cuda_test(l, l, z, o)(4);
    // cuda_test(...) returns CudaChooser&
    // so you can use cuda_test(...).prepare(...).launch()
    // to change execution parameters for the specialization

    cudaDeviceSynchronize();
#endif
}
