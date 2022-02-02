#include "specialut.hpp"
#include <iostream>

// A function with both compile-time and run-time parameters
template<bool i, bool j, int k, bool l>
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

#ifdef USE_CUDA
// A CUDA kernel with both compile-time and run-time parameters
template<bool i, bool j, int k, bool l>
__global__ void cuda_fn(int msg)
{
    int const thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Compile-time specialization: (%i, %i, %i, %i), run-time message: %i\n", i, j, k, l, msg);
}
#endif

int main()
{

    std::cout << "Simple test:" << std::endl;

    SpeciaLUT::Chooser<TABULATE(host_fn), 2, 2, 3, 2> test;
    // instead of the above, shorter macro can be used:
    // CHOOSER(host_fn, 2, 2, 3, 2) test;

    int o = 0, l = 1, z = 2;
    test(o, o, z, l)("Hello #1");
    test(o, l, o, o)("Hey #2");
    test(l, o, l, l)("Hi #3");
    test(l, l, z, o)("Aloha #4");

#ifdef USE_CUDA

    std::cout << "Now doing the CUDA test:" << std::endl;

    SpeciaLUT::CudaChooser<TABULATE(cuda_fn), 2, 2, 3, 2> cuda_test;
    // instead of the above, shorter macro can be used:
    // CUDA_CHOOSER(cuda_fn, 2, 2, 3, 2) cuda_test;

    // set grid size and block size for kernel execution
    cuda_test.prepare({ 1, 1, 1 }, { 1, 1, 1 });
    cuda_test(o, o, z, l)(1);
    cuda_test(o, l, o, o)(2);
    cuda_test(l, o, l, l)(3);
    cuda_test(l, l, z, o)(4);
    // cuda_test(...) returns CudaChooser&
    // so you can use cuda_test(...).prepare(...).launch()
    // to change execution parameters for the specialization

    cudaDeviceSynchronize();
#endif

    return 0;
}
