#include "kernelMethods.cuh"
__device__ unsigned short int *all_sets;
__device__ double costFuncGPU(int R1, int R2){
	return R1*R2;
}

int getMaxThreads(){
	/*
	CUDA DEVICE prop OUT
	###################################
	*/
	int deviceCount, device;
	int gpuDeviceCount = 0;
	int _maxCudaThreads, _maxCudaProcs, _maxCudaShared, _maxSharedPerBlock;
	struct cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess)
		deviceCount = 0;
	/* machines with no GPUs can still report one emulation device */
	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999) /* 9999 means emulation only */
			if (device == 0)
			{
				_maxCudaProcs = properties.multiProcessorCount;
				_maxCudaThreads = properties.maxThreadsPerBlock;
				_maxCudaShared = properties.sharedMemPerMultiprocessor;
				_maxSharedPerBlock = properties.sharedMemPerBlock;
				/*
				printf("multiProcessorCount %d\n", _maxCudaProcs);
				printf("maxThreadsPerMultiProcessor %d\n", _maxCudaThreads);
				printf("maxSharedPerMultiProcessor %d\n", _maxCudaShared);
				printf("max Shared per block %d \n", _maxSharedPerBlock);
				*/
			}
	}
	// ----###########################
	return _maxCudaThreads;
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
__device__ unsigned nextElems(unsigned x)
{
	unsigned smallest, ripple, new_smallest, ones;

	if (x == 0) return 0;
	smallest = (x & -(short int)x);
	ripple = x + smallest;
	new_smallest = (ripple & -(short int)ripple);
	ones = ((new_smallest / smallest) >> 1) - 1;
	return ripple | ones;
}
__global__ void subSetKernel(unsigned short int *table_in, int size_sets, int size_table, int bitSet){

	//int *all_sets;
	if (threadIdx.x == 0){
		//printf("DEVICE array from %d to %d, first value %d \n", 0, size_sets - 1, bitSet);
		//all_sets = new int[size_sets*sizeof(int)];
		all_sets = (unsigned short int*)malloc(size_sets*sizeof(unsigned short int));
		all_sets[0] = bitSet;
		for (int i = 0; i < size_sets - 1; i++){
			bitSet = nextElems(bitSet);
			all_sets[i + 1] = bitSet;
		};
	}
	__syncthreads();

	extern __shared__ unsigned short int shared[];
	unsigned short int * table = (unsigned short int*)&shared;
	unsigned short int idx = threadIdx.x;//+ blockIdx.x*blockDim.x;

	for (unsigned short int i = 0; (i + idx) < size_table; i++){
		table[i + idx] = table_in[i + idx];
	}
	__syncthreads();


	if (idx < size_sets){
		unsigned short int setS, subS1, subS2, cost2;
		unsigned short int  bitS, bitS2;
		bool first_run;
		//Using THREAD INDEX as Array ENtry

		//printf("ALL_S: %d, SET: %d,  for %d \n ", all_sets[idx], sets[idx], idx);
		setS = all_sets[idx];
		bitS = gpuBitCount(setS);
		//calculating the subset S1
		subS1 = setS & (-setS);

		//while the child subset is different from the intiale calcualte the mirror subset
		first_run = true;
		while (subS1 != setS){
			subS2 = setS - subS1;
			bitS2 = gpuBitCount(subS2);
			//for left/right deep tree the subset bit size has to bit 1 less from the set
			if (bitS2 == (bitS - 1)){
				//for intiiale run intiate the set values
				int c1 = table[subS1 * 3];
				int c2 = table[subS2 * 3];
				if (first_run){
					table[setS * 3] = costFuncGPU(c1, c2);
					table[setS * 3 + 1] = subS1;
					table[setS * 3 + 2] = subS2;
					first_run = false;
				}
				else{
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

	}
	for (int i = 0; (i + idx) < size_table; i++){
		table_in[i + idx] = table[i + idx];
	}
}
__global__ void subSetKernel_noShared(unsigned short int *table, int size_sets, int size_table, int bitSet){

	if (threadIdx.x == 0){
		all_sets = (unsigned short int*)malloc(size_sets*sizeof(unsigned short int));
		all_sets[0] = bitSet;
		for (int i = 0; i < size_sets - 1; i++){
			bitSet = nextElems(bitSet);
			all_sets[i + 1] = bitSet;
		};
	}
	__syncthreads();
	unsigned short int idx = threadIdx.x;//+ blockIdx.x*blockDim.x;

	if (idx < size_sets){
		unsigned short int setS, subS1, subS2, cost2;
		unsigned short int  bitS, bitS2;
		bool first_run;
		setS = all_sets[idx];
		bitS = gpuBitCount(setS);
		//calculating the subset S1
		subS1 = setS & (-setS);
		//while the child subset is different from the intiale calcualte the mirror subset
		first_run = true;
		while (subS1 != setS){
			subS2 = setS - subS1;
			bitS2 = gpuBitCount(subS2);
			//for left/right deep tree the subset bit size has to bit 1 less from the set
			if (bitS2 == (bitS - 1)){
				//for intiiale run intiate the set values
				int c1 = table[subS1 * 3];
				int c2 = table[subS2 * 3];
				if (first_run){
					table[setS * 3] = costFuncGPU(c1, c2);
					table[setS * 3 + 1] = subS1;
					table[setS * 3 + 2] = subS2;
					first_run = false;
				}
				else{
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

	}
}
__global__ void subSetKernel_noSets(unsigned short int *table_in, int *sets, int size_sets, int size_table){
	
	extern __shared__ unsigned short int shared[];
	unsigned short int * table = (unsigned short int*)&shared;
	unsigned short int idx = threadIdx.x;//+ blockIdx.x*blockDim.x;

	for (unsigned short int i = 0; (i + idx) < size_table; i++){
		table[i + idx] = table_in[i + idx];
	}
	__syncthreads();


	if (idx < size_sets){
		unsigned short int setS, subS1, subS2, cost2;
		unsigned short int  bitS, bitS2;
		bool first_run;
		setS = sets[idx];
		bitS = gpuBitCount(setS);
		//calculating the subset S1
		subS1 = setS & (-setS);
		//while the child subset is different from the intiale calcualte the mirror subset
		first_run = true;
		while (subS1 != setS){
			subS2 = setS - subS1;
			bitS2 = gpuBitCount(subS2);
			//for left/right deep tree the subset bit size has to bit 1 less from the set
			if (bitS2 == (bitS - 1)){
				//for intiiale run intiate the set values
				int c1 = table[subS1 * 3];
				int c2 = table[subS2 * 3];
				if (first_run){
					table[setS * 3] = costFuncGPU(c1, c2);
					table[setS * 3 + 1] = subS1;
					table[setS * 3 + 2] = subS2;
					first_run = false;
				}
				else{
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

	}
	__syncthreads();
	for (int i = 0; (i + idx) < size_table; i++){
		table_in[i + idx] = table[i + idx];
	}

	__syncthreads();
}
__global__ void subSetKernel_net(unsigned short int *table, int *sets, int size_sets, int size_table, int bitSet){
	
	unsigned short int idx = threadIdx.x;//+ blockIdx.x*blockDim.x;
	if (idx < size_sets){
		unsigned short int setS, subS1, subS2, cost2;
		unsigned short int  bitS, bitS2;
		bool first_run;
		//Using THREAD INDEX as Array ENtry

		//printf("ALL_S: %d, SET: %d,  for %d \n ", all_sets[idx], sets[idx], idx);
		setS = all_sets[idx];
		bitS = gpuBitCount(setS);
		//calculating the subset S1
		subS1 = setS & (-setS);

		//while the child subset is different from the intiale calcualte the mirror subset
		first_run = true;
		while (subS1 != setS){
			subS2 = setS - subS1;
			bitS2 = gpuBitCount(subS2);
			//for left/right deep tree the subset bit size has to bit 1 less from the set
			if (bitS2 == (bitS - 1)){
				//for intiiale run intiate the set values
				int c1 = table[subS1 * 3];
				int c2 = table[subS2 * 3];
				if (first_run){
					table[setS * 3] = costFuncGPU(c1, c2);
					table[setS * 3 + 1] = subS1;
					table[setS * 3 + 2] = subS2;
					first_run = false;
				}
				else{
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

	}
}

//noshared kernel with sets
cudaError_t runOnGpu1(bool *firstGpu, bool *lastGpu, int *firstGpuCoeff, int coeff, int mask, unsigned short int *dp_table, unsigned short int* dev_table, int dp_table_size){

	cudaError_t cudaStatus;
	if (firstGpu){
		cudaStatus = cudaMemcpy(dev_table, dp_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaCopy_in_table failed!");
			goto Error;
		}
		firstGpu = false;
		*firstGpuCoeff = coeff;
	}
	subSetKernel_noShared << <1, coeff >> >(dev_table, coeff, dp_table_size, mask);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KERNEL fail: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	if (lastGpu){
		//Copying back to host
		cudaStatus = cudaMemcpy(dp_table, dev_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaCopy_table FROM DEV failed!");
			goto Error;
		}
	}
Error:
	return cudaStatus;
}

//nosets kernel with shared
cudaError_t runOnGpu2(bool *firstGpu, bool *lastGpu, int *firstGpuCoeff, int coeff, int mask, unsigned short int *dp_table, unsigned short int* dev_table, int dp_table_size, int* all_sets){
	//enumerating subsets on CPU/GPU -> occupancy 
	int *dev_sets;
	cudaError_t cudaStatus;
	//Allocating needed memory 

	cudaStatus = cudaMalloc((void**)&dev_sets, coeff*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_sets, all_sets, coeff*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCopy_in_sets failed!");
		goto Error;
	}

	if (firstGpu){
		cudaStatus = cudaMemcpy(dev_table, dp_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaCopy_in_table failed!");
			goto Error;
		}
		firstGpu = false;
		*firstGpuCoeff = coeff;
	}
	//Launching the kernel
	printf("need to alocate %d \n", dp_table_size*sizeof(unsigned short int));
	subSetKernel_noSets << <1, coeff + 1, dp_table_size*sizeof(unsigned short int) >> >(dev_table, dev_sets, coeff, dp_table_size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KERNEL fail: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	if (lastGpu){
		//Copying back to host
		cudaStatus = cudaMemcpy(dp_table, dev_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaCopy_table FROM DEV failed!");
			goto Error;
		}
	}
	//Freeing cuda allocatade mem
	cudaFree(dev_sets);
Error:
	cudaFree(dev_sets);
	return cudaStatus;
}

//both shared and sets
cudaError_t runOnGpu3(bool *firstGpu, bool *lastGpu, int *firstGpuCoeff, int coeff, int mask, unsigned short int *dp_table, unsigned short int* dev_table, int dp_table_size){

	cudaError_t cudaStatus;
	if (firstGpu){
		cudaStatus = cudaMemcpy(dev_table, dp_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaCopy_in_table failed!");
			goto Error;
		}
		firstGpu = false;
		*firstGpuCoeff = coeff;
	}
	//Launching the kernel
	subSetKernel << <1, coeff + 1, dp_table_size*sizeof(unsigned short int) >> >(dev_table, coeff, dp_table_size, mask);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KERNEL fail: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	if (lastGpu){
		//Copying back to host
		cudaStatus = cudaMemcpy(dp_table, dev_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaCopy_table FROM DEV failed!");
			goto Error;
		}
	}
Error:
	return cudaStatus;
}

//no kernel no sets
cudaError_t runOnGpu4(bool *firstGpu, bool *lastGpu, int *firstGpuCoeff, int coeff, int mask, unsigned short int *dp_table, unsigned short int* dev_table, int dp_table_size, int* all_sets){
	//enumerating subsets on CPU/GPU -> occupancy 
	int *dev_sets;
	cudaError_t cudaStatus;
	//Allocating needed memory 
	cudaStatus = cudaMalloc((void**)&dev_sets, coeff*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_sets, all_sets, coeff*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaCopy_in_sets failed!");
		goto Error;
	}
	if (firstGpu){
		cudaStatus = cudaMemcpy(dev_table, dp_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaCopy_in_table failed!");
			goto Error;
		}
		firstGpu = false;
		*firstGpuCoeff = coeff;
	}
	subSetKernel_net << <1, coeff + 1 >> >(dev_table, all_sets, coeff, dp_table_size, mask);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KERNEL fail: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	if (lastGpu){
		//Copying back to host
		cudaStatus = cudaMemcpy(dp_table, dev_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaCopy_table FROM DEV failed!");
			goto Error;
		}
	}
	cudaFree(dev_sets);
Error:
	return cudaStatus;
}

//enumerating subsets on CPU/GPU -> occupancy 
//Allocating needed memory 
/*
int *dev_sets;
cudaStatus = cudaMalloc((void**)&dev_sets, coeff*sizeof(int));
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaMalloc failed!");
goto Error;
}
cudaStatus = cudaMemcpy(dev_sets, all_sets, coeff*sizeof(int), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaCopy_in_sets failed!");
goto Error;
}
*//*
if (firstGpu){
cudaStatus = cudaMemcpy(dev_table, dp_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyHostToDevice);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaCopy_in_table failed!");
goto Error;
}
firstGpu = false;
firstGpuCoeff = coeff;
}
//Launching the kernel
//subSetKernel <<<1, coeff+1, dp_table_size*sizeof(unsigned short int) >> >(dev_table, coeff, dp_table_size, mask);
subSetKernel_noShared <<<1, coeff + 1>>>(dev_table, coeff, dp_table_size, mask);
cudaStatus = cudaGetLastError();
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "KERNEL fail: %s\n", cudaGetErrorString(cudaStatus));
goto Error;
}
if (lastGpu){
//Copying back to host
cudaStatus = cudaMemcpy(dp_table, dev_table, dp_table_size*sizeof(unsigned short int), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess) {
fprintf(stderr, "cudaCopy_table FROM DEV failed!");
goto Error;
}
}
*/
//Freeing cuda allocatade mem
//cudaFree(dev_sets);