#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

int main() {
	#pragma omp parallel num_threads(3)
    {
        std::cout << "Hello" << std::endl;
    }
    return 0;
}