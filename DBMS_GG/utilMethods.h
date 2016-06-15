#pragma once
#include <stdlib.h>     /* srand, rand */
#include <stdio.h>
#include <Windows.h>
#include <iostream>

void getSubSets(unsigned short int* table, int* sets, int size);
int costFunc(int R1, int R2);
void printResult(unsigned short int *table, int last_node);