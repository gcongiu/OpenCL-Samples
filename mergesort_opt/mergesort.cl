// Size of the local memory buffer
#define BUFFERSIZE 512

// merge sort implementation using global memory
void merge_sort (__global int * input, 
		 __global int * output, 
	         const int threadid, 
		 const int output_size)
{
	// output array boundaries
	int k  = threadid * output_size;

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
	while (i > max_i && j <= max_j)
		output[k++] = input[j++];

	// if the second half is over fill the rest of output with the first half
	while (i <= max_i && j > max_j)
		output[k++] = input[i++];
}

// this is the merge sort implementation for the first 9 sorting stages,
// this implementation exploits local memory (512 elements in size) to sort
// 512 elements array. The global memory is accessed only once at the beginning of
// the algorithm and then all the operations are performed on the local memory
// this implementation works very well and improves performance up to more than 40%
// compared with the previous one
void merge_sort_S1 (__global int * ping,  // unsorted array coming from host
		    __global int * pong,  // intermediate sorted array
		    __local  int * lping, // unsorted local array
		    __local  int * lpong, // intermediate sorted local array
		    __local  int * l_tmp  // temp pointer
		   )
{
	// This kernel is called at the first iteration and generates a 
	// sorted array of 512 elements in size
	
	// local output size and number of thread per work group
	int l_output_size;
	int l_num_thread;

	// at the beginning the algorithm combine single elements in two elements
	// sorted arrays
	l_output_size = 2;

	// this parameter is used since the number of threads is halfed at 
	// every iteration
	l_num_thread = get_local_size(0);
	
	// read the data from global memory
	lping[get_local_id(0) * l_output_size] = 
			ping[get_group_id(0) * BUFFERSIZE + get_local_id(0) * l_output_size];
	lping[get_local_id(0) * l_output_size + 1] = 
			ping[get_group_id(0) * BUFFERSIZE + get_local_id(0) * l_output_size + 1];

	// wait for the buffer lping to be completely full
	barrier(CLK_LOCAL_MEM_FENCE);

	int i, j, k, max_i, max_j;

	// iterate until the local buffer is full sorted
	for (; l_output_size <= get_local_size(0) * 2; l_num_thread /=2, l_output_size *=2) 
	{
		// output array boundaries		
		k = get_local_id(0) * l_output_size;
	
		// input array boundaries
		i = k;
		max_i = i + l_output_size/2 - 1;
		j = max_i + 1;
		max_j = k + l_output_size - 1;

		if (get_local_id(0) < l_num_thread)
		{
			// compare the first half with the second half
			while (i <= max_i && j <= max_j)
			{
				if (lping[i] <= lping[j])
					lpong[k++] = lping[i++];
				else
					lpong[k++] = lping[j++];
			}
		
			// if the first half is over fill the rest of output with the second half
			while (i > max_i && j <= max_j)
				lpong[k++] = lping[j++];
		
			// if the second half is over fill the rest of output with the first half
			while (i <= max_i && j > max_j)
				lpong[k++] = lping[i++];
		}	

		// update local variables for next iteration
		l_tmp = lping;
		lping = lpong;
		lpong = l_tmp;
		 
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// copy local output buffer to global output buffer
	pong[get_group_id(0) * BUFFERSIZE + get_local_id(0) * 2]     = lping[get_local_id(0) * 2];
	pong[get_group_id(0) * BUFFERSIZE + get_local_id(0) * 2 + 1] = lping[get_local_id(0) * 2 + 1];
}

// this merge sort implementation takes care of sorting the 512*N elements
// arrays into a 1024*N buffer, if N = 1 then the 512 element arrays are
// the output of the previous implementation
// this implementation works very bad: only 1 thread per group seems to be not 
// a very good idea, even when using local memory performance are really bad
void merge_sort_SN (__global int * ping,  // unsorted array coming from host
		    __global int * pong,  // intermediate sorted array
		    __local  int * lping, // unsorted local array
		    __local  int * lpong, // intermediate sorted local array
		    const int iteration   // current iteration number
		   )
{
	// here the num of thread per group is 1
	int l_output_size = (int)(1 << iteration); // size of the output array
	int l_input_size = l_output_size/2; 	   // size of the input arrays

	// divide the two input buffer into sub-buffer of size BUFFERSIZE
	int offset_i = 0, offset_j = 0, offset_k = 0;
	
	// fill in the local buffers
	int n;
	for (n = 0; n < BUFFERSIZE/2; n++)
	{
		lping[n] = 
			ping[get_group_id(0) * l_output_size + offset_i + n];
		lping[BUFFERSIZE/2 + n] =
		        ping[get_group_id(0) * l_output_size + l_output_size/2 + offset_j + n];
	}

	int i, j, k, max_i, max_j, max_k;

	// output array boundaries		
	k = 0;
	max_k = BUFFERSIZE - 1;

	// input array boundaries
	i = 0;
	max_i = BUFFERSIZE/2 - 1;
	j = BUFFERSIZE/2;
	max_j = BUFFERSIZE - 1;

	while (offset_i < l_input_size && 
	       offset_j < l_input_size   ) 
	{
		// compare the first half with the second half
		while (i <= max_i && j <= max_j && k <= max_k)
		{
			if (lping[i] <= lping[j])
				lpong[k++] = lping[i++];
			else
				lpong[k++] = lping[j++];
		}
		
		// if i runs out of elements update i
		if (i > max_i)
		{
			offset_i += BUFFERSIZE/2;
			// update 
			for (n = 0; n < BUFFERSIZE/2 && offset_i < l_input_size; n++)
			{
				lping[n] = ping[get_group_id(0) * l_output_size +
					          offset_i + n];
			}
			i = 0;
		}

		// if j runs out of elements update j
		if (j > max_j)
		{
			offset_j += BUFFERSIZE/2;
			// update l_input with new data from g_input
		 	for (n = 0; n < BUFFERSIZE/2 && offset_j < l_input_size; n++)
			{
				lping[BUFFERSIZE/2 + n] = ping[get_group_id(0) * l_output_size +
					             		 l_input_size + 
								 offset_j + n];
			}
			j = BUFFERSIZE/2;
		}

		// if the l_output buffer gets full drain it
		if (k > max_k)
		{
			for (n = 0; n < BUFFERSIZE; n++)
			{
				pong[get_group_id(0) * l_output_size + offset_k + n] = lpong[n];
			}
			k = 0;
			offset_k += BUFFERSIZE;
		}
	}

	while (offset_j < l_input_size)
	{
		while (j <= max_j && k <= max_k)
			lpong[k++] = lping[j++];

		// if j runs out of elements update j
		if (j > max_j)
		{
			offset_j += BUFFERSIZE/2;
			// update l_input with new data from g_input
		 	for (n = 0; n < BUFFERSIZE/2 && offset_j < l_input_size; n++)
			{
				lping[BUFFERSIZE/2 + n] = ping[get_group_id(0) * l_output_size +
					             		 l_input_size + 
								 offset_j + n];
			}
			j = BUFFERSIZE/2;
		}

		// if the l_output buffer gets full drain it
		if (k > max_k)
		{
			for (n = 0; n < BUFFERSIZE; n++)
			{
				pong[get_group_id(0) * l_output_size + offset_k + n] = lpong[n];
			}
			k = 0;
			offset_k += BUFFERSIZE;
		}
	}
	
	while (offset_i < l_input_size)
	{ 
		while (i <= max_i && k <= max_k)
			lpong[k++] = lping[i++];
	
		// if i runs out of elements update i
		if (i > max_i)
		{
			offset_i += BUFFERSIZE/2;
			// update l_input with new data from g_input
			for (n = 0; n < BUFFERSIZE/2 && offset_i < l_input_size; n++)
			{
				lping[n] = ping[get_group_id(0) * l_output_size +
					          offset_i + n];
			}
			i = 0;
		}

		// if the l_output buffer gets full drain it
		if (k > max_k)
		{
			for (n = 0; n < BUFFERSIZE; n++)
			{
				pong[get_group_id(0) * l_output_size + offset_k + n] = lpong[n];
			}
			k = 0;
			offset_k += BUFFERSIZE;
		}
	
	}	
}

// this is the kernel which is called by the host code
__kernel
void _sort (__global int * ping,   // input/output array 
	    __global int * pong,   // output array
	    const int size,        // size of the array to sort
	    const int iteration    // sorting stage starts from 1
	   )  
{
	// local arrays used by every workgroup 
	__local int lping[BUFFERSIZE];
	__local int lpong[BUFFERSIZE];
	__local int * ltmp;

	// thread and group ids
	unsigned int groupid    = get_group_id(0);    // group identifier
	unsigned int threadid   = get_local_id(0);    // thread identifier within the group
	unsigned int localsize  = get_local_size(0);  // number of threads within the group
	unsigned int globalsize = get_global_size(0); // total number of threads

	unsigned int output_size= (1 << iteration);

	if (iteration == 1)
	{
		merge_sort_S1 (ping,
			       pong,
			       lping,
			       lpong,
			       ltmp
			      );
	}
	else
#if 0
		// if the sorting stage is even input and output are 
		// swaped: sort input->output and then sort output->input
		if ( (iteration % 2) == 0)
			merge_sort (pong,
				    ping,
				    threadid,
				    output_size
				   );
		else
			merge_sort (ping,
				    pong,
				    threadid,
				    output_size
				   );
#else
		if ( (iteration % 2) == 0 )
		{
			merge_sort_SN (pong, 
				       ping, 
				       lping,
				       lpong,
				       iteration);
		}
		else
		{
			merge_sort_SN (ping, 
				       pong, 
				       lping,
				       lpong,
				       iteration);
		}
#endif
}
