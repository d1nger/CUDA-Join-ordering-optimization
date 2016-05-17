#include "utilMethods.h"
#include "binMethod.h"
#include <stdlib.h>     /* srand, rand */
#include <stdio.h>
void getSubSets(int* table, int* sets, int size){
	int setS, subS1, subS2, cost2;
	int bitS, bitS2;
	bool first_run;
	//For every elemnt of the erray : every set S
	for (int k = 0; k < size; k++){
		setS = sets[k];
		bitS = countSetBits(setS);
		//calculating the subset S1
		subS1 = setS & (-setS);
		//while the child subset is different from the intiale calcualte the mirror subset
		first_run = true;
		while (subS1 != setS){
			subS2 = setS - subS1;
			bitS2 = countSetBits(subS2);
			//for left/right deeop tree the subset bit size has to bit 1 less from the set
			if (bitS2 == (bitS - 1)){
				//for intiiale run intiate the set values
				int c1 = table[subS1 * 3];
				int c2 = table[subS2 * 3];
				if (first_run){
					table[setS * 3] = costFunc(c1, c2);
					  table[setS * 3 + 1] = subS1;
					  table[setS * 3 + 2] = subS2;
					  first_run = false;
				}else{
					//once better path found set this one to the dp table
					cost2 = costFunc(c1, c2);
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

int costFunc(int R1, int R2)
{
	double cost = R1 * R2 * (rand()%300) /300;
	return (int) cost;
}

void printResult(int *table, int last_node){
	printf("Best join Right-Deep-Tree is : \n");
	int i = last_node;
	int node_size = countSetBits(i);
	while (node_size > 1){
		printf(" %d x ", table[i * 3 + 1]);
		i = table[i * 3 + 2];
		node_size = countSetBits(i);
	}
	printf(" %d \n", i);
}