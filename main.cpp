#include "specialut.hpp"
#include <iostream>

// A function with both compile-time and run-time parameters
template<bool i, bool j, int k, bool l>
void host_fn(const char* message)
{
    std::cout << "Compile-time specialization: (" << i << ", " << j << ", " << k << ", " << l
              << "), run-time message: " << message << std::endl;
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

    FUNCTION_CHOOSER(host_fn, 2, 2, 3, 2) test;

    int o = 0, l = 1, z = 2;
    test(o, o, z, l)("Hi");
    test(o, l, o, o)("this");
    test(l, o, l, l)("is");
    test(l, l, z, o)("test");

#ifdef USE_CUDA

    std::cout << "Now doing the CUDA test:" << std::endl;

    CUDA_CHOOSER(cuda_fn, 2, 2, 3, 2) cuda_test;

    cuda_test.prepare({ 1, 1, 1 }, { 1, 1, 1 });
    cuda_test(o, o, z, l)(1);
    cuda_test(o, l, o, o)(2);
    cuda_test(l, o, l, l)(3);
    cuda_test(l, l, z, o)(4);

    cudaDeviceSynchronize();
#endif

    return 0;
}
