#include "binMethod.h"

int countSetBits(unsigned int n)
{
	unsigned int count = 0;
	while (n)
	{
		count += n & 1;
		n >>= 1;
	}
	return count;
}

int getTableIndex(int table_bit)
{
	int count = 1;
	while (table_bit != 0) {
		if ((table_bit & 0x1) == 1)
			break;
		table_bit = table_bit >> 0x01;
		count++;
	}

	return count;
}

unsigned next_set_of_n_elements(unsigned x)
{
	unsigned smallest, ripple, new_smallest, ones;

	if (x == 0) return 0;
	smallest = (x & -(int)x);
	ripple = x + smallest;
	new_smallest = (ripple & -(int)ripple);
	ones = ((new_smallest / smallest) >> 1) - 1;
	return ripple | ones;
}

int binomialeCoeff(int num, int max)
{
	/*
	tgamma =  factorial de n+1
	Cnk = k! / (k-n)!
	*/
	int coeff = tgamma(max + 1.0) / (tgamma(max - num + 1.0)*tgamma(num + 1.0));
	return coeff;
}