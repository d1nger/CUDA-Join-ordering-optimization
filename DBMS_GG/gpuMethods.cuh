#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <math.h>
__device__ double costFuncGPU(double R1, double R2, double sel_12);

__global__ void subSetKernel();

int* getSubSets(int bits, int max);