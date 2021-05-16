#include <iostream>
#include "specialut.hpp"

struct Test
{
    template<bool i, bool j, int k, bool l>
    static void run(const char* message)
    {
        std::cout << "Specialization: (" << i << "," << j << "," << k << "," << l << ") message: " << message << std::endl;
    }
};


int main()
{

    SpeciaLUT::Chooser<Test, 2, 2, 3, 2> test;

    int o = 0, l = 1, z = 2;
    test(o,o,z,l)("Hi");
    test(o,l,o,o)("this");
    test(l,o,l,l)("is");
    test(l,l,z,o)("test");

    return 0;
}


