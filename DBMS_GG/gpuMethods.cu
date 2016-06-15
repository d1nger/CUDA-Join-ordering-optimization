#include "binMethod.h"
#include "utilMethods.h"
#include "kernelMethods.cuh"

#define DP_SIZE  12288
#define MIN(a,b) ((a) < (b) ? (a) : (b))

double PCFreq;
__int64 CounterStart;

double average(double numbers[], int size);
void StartCounter();
double getCounter();

double runSimulation(){
			PCFreq = 0.0;
			CounterStart = 0;
			//Get max threads on device
			int MaxTHreads = getMaxThreads();
			int _getMaxTHreads = MaxTHreads;
			unsigned short int input = 0x7FFF;
			int input_count = countSetBits(input);

			//input tables
			int* sql_input = new int[input_count];
			for (int i = 0; i <= input_count; i++){
				sql_input[i] = rand() % 300 + 1;
			}
			//double sql_sel[5] = { 0.01, 0.34, 0.55, 0.28, 0.88 };
			int bitNumber = countSetBits(input);
			int dp_table_size = 3 * (int)pow(2.0, bitNumber);


			//START OF LOOP
			unsigned short int* dp_table = new unsigned short int[dp_table_size];
			cudaError_t cudaStatus;
			// Allocate GPU buffer
			int *dev_sel;
			unsigned short int *dev_table;
			cudaStatus = cudaMalloc((void**)&dev_sel, 5 * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				goto Error;
			}
			cudaStatus = cudaMalloc((void**)&dev_table, dp_table_size*sizeof(unsigned short int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				goto Error;
			}

			//Levels enumeration
			int mask = 0x01;
			int currBitSet, coeff, bitSet, table, min;
			bool firstGpu = true, lastGpu = false;
			int firstGpuCoeff = 0;
			clock_t start_t = clock();
			StartCounter();
			for (int j = 1; j <= bitNumber; j++) {
				bitSet = mask;
				currBitSet = countSetBits(bitSet);
				//setting up the first level = importing the initale tables caridnality
				if (currBitSet == 1){
					table = getTableIndex(currBitSet);
					dp_table[currBitSet * 3] = sql_input[table];
					for (int i = 0; i < bitNumber - 1; i++){
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
					//printf("Initiale array from %d to %d, first value %d \n", 0, coeff -1,bitSet);
					int *host_sets = new int[coeff*sizeof(int)];
					host_sets[0] = bitSet;
					for (int i = 0; i < coeff - 1; i++){
						bitSet = next_set_of_n_elements(bitSet);
						host_sets[i + 1] = bitSet;
					};
					//printf("Binom Coefficient is: %d \n", coeff);
					if (coeff == firstGpuCoeff){
						lastGpu = true;
					}
					if (coeff  > 16){
						min = MIN(coeff, _getMaxTHreads);
						cudaStatus = runOnGpu1(&firstGpu, &lastGpu, &firstGpuCoeff, min, mask, dp_table, dev_table, dp_table_size);
					}
					else {
						//running the subset enumaration on CPU
						getSubSets(dp_table, host_sets, coeff);
					}

					delete[] host_sets;
					host_sets = NULL;
				}

				mask = mask << 1;
				mask = mask + 1;
			}
			//printf("Time taken: %.2fs\n", (double)(clock() - start_t) / CLOCKS_PER_SEC);
			printf("Time taken: %f \n", getCounter());
			cudaFree(dev_table);
			cudaFree(dev_sel);

			//printResult(dp_table, input);

			free(dp_table);
			//END OF LOOP
			return  getCounter();

		Error:
			cudaFree(dev_table);
			cudaFree(dev_sel);
			return 0;
};

int main(){
	int const number_of_tests = 10;
	double timeArray[number_of_tests];
	for (int k = 0; k < number_of_tests; k++){
		timeArray[k] = runSimulation();
	}
	double avg = average(timeArray, number_of_tests);
	printf("AVG time: %f \n", avg);
}

void StartCounter(){
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed! \n";

	PCFreq = double(li.QuadPart);// / 1000000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double getCounter(){
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}
double average(double numbers[], int size) {
	double sum = 0;
	for (int x = 0; x < size; x++)
	{
		sum += numbers[x];
	}
	return sum / (double)size;
}