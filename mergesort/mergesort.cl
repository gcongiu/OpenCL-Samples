void merge_sort (__global int * input, 
		 __global int * output, 
		 const int output_size
		)
{
	// output array boundaries
	int k  = get_global_id(0) * output_size;

	// input array boundaries
	int i     = k;
	int max_i = i + output_size/2 - 1;
	int j     = max_i + 1;
	int max_j = k + output_size - 1;

	// compare the first half with the second half
	while (i <= max_i && j <= max_j)
	{
		if (input[i] <= input[j])
			output[k++] = input[i++];
		else
			output[k++] = input[j++];
	}

	// if the first half is over fill the rest of output with the second half
	while (j <= max_j)
		output[k++] = input[j++];

	// if the second half is over fill the rest of output with the first half
	while (i <= max_i)
		output[k++] = input[i++];
}

// sorting kernel - ping is the input vector and depending on the number of sorting stages
// can also be the output vector
__kernel
void _sort (__global int * ping,
	    __global int * pong, 
	    const int output_size, 
	    const int iteration
	   )
{
	// during the first iteration
	// read the input array from input 
	// and store the partially sorted 
	// array in output. during the second
	// iteration invert, read from output
	// and store in input
	if ( (iteration % 2) == 0)
	{
		merge_sort (pong, ping, output_size);
	}
	else
	{
		merge_sort (ping, pong, output_size);
	}
}
