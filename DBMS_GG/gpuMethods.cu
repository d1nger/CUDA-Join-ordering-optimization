#include "binMethod.h"
#include "utilMethods.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <math.h>

#define DP_SIZE  12288
__device__ double costFuncGPU(int R1, int R2){
	return R1*R2;
}
__device__ int gpuBitCount(unsigned int n){
	unsigned int count = 0;
	while (n)
	{
		count += n & 1;
		n >>= 1;
	}
	return count;
}
__global__ void subSetKernel(int *table, int *sets, int size){

	if (threadIdx.x == 0){

	}
	else if(threadIdx.x < size +1){
		int setS, subS1, subS2, cost2;
		int bitS, bitS2;
		bool first_run;
		//Using THREAD INDEX as Array ENtry
		setS = sets[threadIdx.x-1];

		bitS = gpuBitCount(setS);
		//calculating the subset S1
		subS1 = setS & (-setS);

		//while the child subset is different from the intiale calcualte the mirror subset
		first_run = true;
		while (subS1 != setS){
			subS2 = setS - subS1;
			bitS2 = gpuBitCount(subS2);
			//for left/right deeop tree the subset bit size has to bit 1 less from the set
			if (bitS2 == (bitS - 1)){
				//for intiiale run intiate the set values
				int c1 = table[subS1 * 3];
				int c2 = table[subS2 * 3];
				printf("Cost for %d is %d, and for %d is %d \n", subS1, c1, subS2, c2);
				if (first_run){
					table[setS * 3] = costFuncGPU(c1, c2);
					  table[setS * 3 + 1] = subS1;
					  table[setS * 3 + 2] = subS2;
					  first_run = false;
				}else{
					//once better path found set this one to the dp table
					cost2 = costFuncGPU(c1, c2);
					if (table[setS] > cost2){
						table[setS * 3] = cost2;
						table[setS * 3 + 1] = subS1;
						table[setS * 3 + 2] = subS2;
					}
				}
			}
			subS1 = setS & (subS1 - setS);
		}

		printf("Best_Cost is %d for %d \n", table[setS*3], setS);
	}
	__syncthreads();
}

cudaError_t runOnGpu();

int main(){

	int input = 0x0F;
	int input_count = countSetBits(input);

	//input tables
	int* sql_input = new int[input_count];
	for (int i = 0; i <= input_count; i++){
		sql_input[i] = rand() % 300 + 1;
	}
	//double sql_sel[5] = { 0.01, 0.34, 0.55, 0.28, 0.88 };
	int bitNumber = countSetBits(input);
	int dp_table_size = 3 * (int)pow(2.0, bitNumber );
	printf("INPUT NUMBER  = %d, dp size = %d \n", bitNumber, dp_table_size);
	//int* dp_table = new int[dp_table_size];
	int dp_table[DP_SIZE];
	// [SIZE ][CHILD_1 ][CHILD_2];

	cudaError_t cudaStatus;
	// Allocate GPU buffer
	int *dev_sel;
	 int*dev_table;
	cudaStatus = cudaMalloc((void**)&dev_sel, 5 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_table, dp_table_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	//Levels enumeration
	int mask = 0x01;
	int currBitSet, coeff, bitSet, table;
	printf("BITNUMBER: %d \n", bitNumber);
	for (int j = 1; j <= bitNumber; j++) {
		bitSet = mask;
		currBitSet = countSetBits(bitSet);
		printf("\n LVL IS : %d \n", j);
		//setting up the first level = importing the initale tables caridnality
		if (currBitSet == 1){
			table = getTableIndex(currBitSet);
			dp_table[currBitSet * 3] = sql_input[table];
			for (int i = 0; i < bitNumber -1; i++){
				currBitSet = next_set_of_n_elements(currBitSet);
				table = getTableIndex(currBitSet);
				dp_table[currBitSet * 3] = sql_input[table];
			}
		}
		else{
			//getting the coefficient
			if (currBitSet != bitNumber){
				coeff = binomialeCoeff(currBitSet, bitNumber);
			}
			else{
				coeff = 1;
			};

			//getting array of all sets to treat;
			int *all_sets = new int[coeff*sizeof(int)];
			all_sets[0] = bitSet;
			for (int i = 0; i < coeff - 1; i++){
				bitSet = next_set_of_n_elements(bitSet);
				all_sets[i + 1] = bitSet;
			};
			//enumerating subsets on CPU/GPU -> resolving occupancy problem
			if (coeff > 400){
//running the subset enumaration on GPU
				int *dev_sets;
//Allocating needed memory 
				cudaMalloc((void**)&dev_sets, coeff*sizeof(int));
				cudaMemcpy(dev_sets, all_sets, coeff*sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_table, dp_table, DP_SIZE*sizeof(int), cudaMemcpyHostToDevice);
//Launching the kernel
				subSetKernel <<<1, coeff+1>>>(dev_table, dev_sets, coeff);
//Copying back to host
				cudaMemcpy(dp_table, dev_table, DP_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//Freeing cuda allocatade mem
				cudaFree(dev_sets);

			}
			else {
				//running the subset enumaration on CPU
				getSubSets(dp_table, all_sets, coeff);
			}
			delete[] all_sets;
			all_sets = NULL;
		}
		mask = mask << 1;
		mask = mask + 1;
	}
	cudaFree(dev_table);
	for (int l = 1; l < input; l++){
		if (countSetBits(l) == 1){
			printf("TABLE is %d \n", dp_table[l * 3]);
		}
		else{
			printf("TABLE is %d and %d \n", dp_table[l * 3 + 1], dp_table[l * 3 + 2]);
		}
	}

Error:
	return cudaStatus;
}


cudaError_t runOnGpu();