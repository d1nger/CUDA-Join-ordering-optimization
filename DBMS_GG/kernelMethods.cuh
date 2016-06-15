#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <string.h>
#include <math.h>
#include <time.h>

int getMaxThreads();
__device__ double costFuncGPU(double R1, double R2, double sel_12);
__device__ int gpuBitCount(unsigned int n);
__device__ unsigned nextElems(unsigned x);
__global__ void subSetKernel(unsigned short int *table_in, int size_sets, int size_table, int bitSet);
__global__ void subSetKernel_noShared(unsigned short int *table_in, int size_sets, int size_table, int bitSet);
__global__ void subSetKernel_noSets(unsigned short int *table_in, int *sets, int size_sets, int size_table);
__global__ void subSetKernel_net(unsigned short int *table, int *sets, int size_sets, int size_table, int bitSet);
cudaError_t runOnGpu1(bool *firstGpu, bool *lastGpu, int *firstGpuCoeff, int coeff, int mask, unsigned short int *dp_table, unsigned short int* dev_table, int dp_table_size);
cudaError_t runOnGpu2(bool *firstGpu, bool *lastGpu, int *firstGpuCoeff, int coeff, int mask, unsigned short int *dp_table, unsigned short int* dev_table, int dp_table_size, int* all_sets);
cudaError_t runOnGpu3(bool *firstGpu, bool *lastGpu, int *firstGpuCoeff, int coeff, int mask, unsigned short int *dp_table, unsigned short int* dev_table, int dp_table_size);
cudaError_t runOnGpu4(bool *firstGpu, bool *lastGpu, int *firstGpuCoeff, int coeff, int mask, unsigned short int *dp_table, unsigned short int* dev_table, int dp_table_size, int* all_sets);