// System includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// OpenCL includes
#include "CL/opencl.h"

// The size of the buffer used by the OpenCL kernel -
// the size of the buffer also set the maximum number
// of sorting threads: 512/2 = 256

#define BUFFERSIZE 512

// Check the status returned by the OpenCL API functions
#define CHECK(status) 								\
	if (status != CL_SUCCESS)						\
	{									\
		fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
		exit(1);							\
	}									\

// Naive sort implementation => complexity O(n²)
void naive_sort(unsigned int * vector, size_t size)
{
	unsigned int i, j, tmp;

	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			if (vector[i] < vector[j])
			{
				tmp = vector[i];
				vector[i] = vector[j];
				vector[j] = tmp;
			}
		}
	}

#ifdef _ENABLE_DEBUG_NV_
	for (i = 0; i < size; i++)
		fprintf(stderr, "%u ", vector[i]);
	fprintf(stderr, "\n");
#endif
}

// merge sort implementation => complexity O(n·log2(n))
void merge_sort(unsigned int* vector, size_t size)
{
	// define indexes
	unsigned int k, i, j;

	unsigned int *low, *high;

	// unsorted vector pointers
	low  = vector;
	high = &(*(vector + size/2));

#ifdef _ENABLE_DEBUG_MS_
	printf("\nsize: %u\n", (int)size);
#endif
	// temp vector to store the sorted values
	unsigned int *tmp = (unsigned int*)malloc(sizeof(int) * size);

	// devide the problem until compare 2 single element vectors (recursion)
	if (size > 2)
	{
		// apply recursively the algorithm to the first half
		(void)merge_sort(low, size/2);
		// apply recursively the algorithm to the second half
		(void)merge_sort(high, size/2);
	}

	// initialise indexes
	i = 0;
	j = size/2;

	unsigned int max_i = size/2 - 1;
	unsigned int max_j = size - 1;

	k = 0;
	// compare the first half with the second half
	while (i <= max_i && j <= max_j)
	{
		if (vector[i] <= vector[j])
			tmp[k++] = vector[i++];
		else
			tmp[k++] = vector[j++];
	}

	// if the first half is over fill the rest of tmp with the second half
	while (i > max_i && j <= max_j)
		tmp[k++] = vector[j++];

	// if the second half is over fill the rest of tmp with the first half
	while (j > max_j && i <= max_i)
		tmp[k++] = vector[i++];

	// copy the sorted values in the original vector
	for (k = 0; k < size; k++)
	{
		vector[k] = tmp[k];
#ifdef _ENABLE_DEBUG_MS_
		printf("%u ", vector[k]);
#endif
	}

#ifdef _ENABLE_DEBUG_MS_
	printf("\n");
#endif
	free(tmp);

	return;
}

int main(int argc, char* argv[]) {

	// check inputs
	if (argc != 2)
	{
		printf("\nUsage:  %s <vector size>\n\n", argv[0]);
		exit(1);
	}

	// vector size
	size_t elements = atoi(argv[1]);

	FILE * f_out = stdout;

	// for simplicity we consider vector sizes to be power of 2
	if ( elements != pow(2, (unsigned int)ceil(log2(elements))) )
	{
		elements = pow(2, (unsigned int)log2(elements));
		printf("the vector size must be a power of 2.\n");
		printf("the vector size has been reset to %u\n", (unsigned int)elements);
	}

	// Timings
	struct timeval start_time, stop_time;
	time_t cl_elapsed, naive_elapsed, cpu_elapsed;

        // OpenCL profiling info
	cl_event timing_event;
	cl_int err_code;
	cl_ulong starttime, endtime, cumulative = 0;
	unsigned long kernel_elapsed;

	// Start of the OpenCL code section

	// Host data
	unsigned int *X = NULL; // Input/Output array to/from accelerator
	unsigned int *Y = NULL; // Output array from the accelerator
	unsigned int *Z = NULL; // naive sort array (for comparison)
	unsigned int *K = NULL; // CPU merge sort array (for comparison)

	// Compute the size of the data
	size_t datasize = sizeof(int)*elements;

	// Allocate space for input/output data
	X = (unsigned int *) malloc (datasize);
	Y = (unsigned int *) malloc (datasize);
	Z = (unsigned int *) malloc (datasize);
	K = (unsigned int *) malloc (datasize);


	// Initialize the input data
	int i;
#ifdef _ENABLE_DEBUG_
	fprintf(stderr, "unsorted array:\n");
#endif
	for (i = 0; i < elements; i++)
	{
		X[i] = (unsigned int)rand()%100;
		Z[i] = X[i];
		K[i] = Z[i];
#ifdef _ENABLE_DEBUG_
		fprintf(stderr, "%u ", X[i]);
#endif
	}
#ifdef _ENABLE_DEBUG_
	fprintf(stderr,"\n");
#endif
	// Use this to check the output of each API call
	cl_int status;

//----------------------------------------------
// STEP 1: Discover and initialize the platforms
//----------------------------------------------
	cl_uint numPlatforms = 0;
	cl_platform_id* platforms = NULL;

	// Use clGetPlatformIDs() to retrieve the
	// number of platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);

	// Allocate enough space for each platform
	platforms = (cl_platform_id*)
		malloc (numPlatforms * sizeof(cl_platform_id));

	// Fill in platforms with clGetPlatformIDs()
	status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK(status);

//----------------------------------------------
// STEP 2: Discover and initialize the devices
//----------------------------------------------

	cl_uint numDevices = 0;
	cl_device_id* devices = NULL;
	// Use clGetDeviceIDs() to retrieve the number of
	// devices present
	status = clGetDeviceIDs(platforms[0],
				CL_DEVICE_TYPE_ALL,
				0,
				NULL,
				&numDevices); CHECK(status);

	// Allocate enough space for each device
	devices = (cl_device_id*)
		malloc (numDevices * sizeof(cl_device_id));

	// Fill in devices with clGetDeviceIDs()
	status = clGetDeviceIDs(platforms[0],
				CL_DEVICE_TYPE_ALL,
				numDevices,
				devices,
				NULL);

	// Device info
	char buffer[4096];
	unsigned int buf_uint;
	for (i = 0; i < numDevices; i++)
	{
		clGetDeviceInfo(devices[i],
				CL_DEVICE_NAME,
				4096,
				buffer,
				NULL);
		fprintf(f_out, "\nDevice Name: %s\n", buffer);

		clGetDeviceInfo(devices[i],
				CL_DEVICE_VENDOR,
				4096,
				buffer,
				NULL);
		fprintf(f_out, "Device Vendor: %s\n", buffer);

		clGetDeviceInfo(devices[i],
				CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(buf_uint),
				&buf_uint,
				NULL);
		fprintf(f_out, "Device Computing Units: %u\n\n", buf_uint);
	}


//----------------------------------------------
// STEP 3: Create a context
//----------------------------------------------

	cl_context context = NULL;

	// Create a context using clCreateContext() and
	// associate it with the device

	context = clCreateContext(
			NULL,
			1,
			devices,
			NULL,
			NULL,
			&status); CHECK(status);

//----------------------------------------------
// STEP 4: Create a command queue
//---------------------------------------------

	cl_command_queue cmdQueue;
	// Create a command queue using clCreateCommandQueue(),
	// and associate it with the device you want to execute
	// on
	cmdQueue = clCreateCommandQueue(
			context,
			devices[0],
			CL_QUEUE_PROFILING_ENABLE,
			&status); CHECK(status);

//----------------------------------------------
// STEP 5: Create device buffers
//----------------------------------------------

	cl_mem bufferX; // Input/Output array on the device
	cl_mem bufferY; // Output array on the device

	// Use clCreateBuffer() to create a buffer object (d_X)
	// that will contain the data from the host array X
	bufferX = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE,
			datasize,
			NULL,
			&status); CHECK(status);

	// Use clCreateBuffer() to create a buffer object (d_Y)
	// that will contain the data from the host array Y
	bufferY = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE,
			datasize,
			NULL,
			&status); CHECK(status);

//----------------------------------------------
// STEP 6: Write host data to device buffers
//----------------------------------------------

	// Use clEnqueueWriteBuffer() to write input array X to
	// the device buffer bufferX
	status = clEnqueueWriteBuffer(
			cmdQueue,
			bufferX,
			CL_FALSE,
			0,
			datasize,
			X,
			0,
			NULL,
			NULL); CHECK(status);

//----------------------------------------------
// STEP 7: Create the program from binaries
//----------------------------------------------

	// Open the file containing the kernel binary
	FILE * fp = fopen("mergesort.ptx", "rb");
	fseek(fp, 0, SEEK_END);
	size_t binarysize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// Allocate enough space to contain the binary code
	unsigned char * binary = (unsigned char *) malloc (binarysize);

	// Read the binary code from the binary file
	size_t dataread = fread(binary, 1, binarysize, fp);
	if (dataread == 0)
	{
		fprintf(stderr, "error in line %d: reading binary from file\n", __LINE__);
		exit(1);
	}

	// Create a program using clCreateProgramWithBinary()
	cl_program program = clCreateProgramWithBinary(
			context,
			1,
			devices,
			&binarysize,
			(const unsigned char **)&binary,
			&status,
			NULL); CHECK(status);

//----------------------------------------------
// STEP 8: Create the kernel
//----------------------------------------------

	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	cl_kernel kernel = NULL;
	// Set the kernel name - this is the name of the kernel function
	// within the cl file
	const char *kernel_name = "_sort";

	// Use clCreateKernel() to create a kernel
	kernel = clCreateKernel(program, (const char*)kernel_name, &status); CHECK(status);

	// compute the number of sorting stage: iterations
	unsigned int iterations = (unsigned int)log2(elements);

	// size of the sorted array
	unsigned int output_size = elements;

	// iteration number: start with 1 of course
	i = 1;

	// start the timer
	(void)gettimeofday(&start_time, NULL);
	// Associate the input and output buffers with the
	// kernel
	// using clSetKernelArg()
	status = clSetKernelArg(
			kernel,
			0,
			sizeof(cl_mem) ,
			(void*)&bufferX); CHECK(status);

	status |= clSetKernelArg(
			kernel,
			1,
			sizeof(cl_mem),
			(void*)&bufferY); CHECK(status);

	status |= clSetKernelArg(
			kernel,
			2,
			sizeof(unsigned int),
			(void*)&output_size); CHECK(status);

	status |= clSetKernelArg(
			kernel,
			3,
			sizeof(unsigned int),
			(void*)&i); CHECK(status);

//----------------------------------------------
// STEP 10: Configure the work-item structure
//----------------------------------------------

	// Define the number of threads that will be created
	// as well as the number of work groups
	size_t globalWorkSize[1];
	size_t localWorkSize[1];

	// At the first iteration the number of threads is elements/2
	// since we start sorting single element arrays into two elements
	// arrays
	globalWorkSize[0] = elements/2;

	// The number of threads per group can be either equal to 256 if the
	// number of elements exceeds the BUFFERSIZE or elements/2 if it is
	// lower than BUFFERSIZE
	localWorkSize[0]  = (elements >= BUFFERSIZE) ? BUFFERSIZE/2 : elements/2;

//----------------------------------------------
// STEP 11: Enqueue the kernel for execution
//----------------------------------------------

	// Execute the kernel by using
	// clEnqueueNDRangeKernel().
	status = clEnqueueNDRangeKernel(
			cmdQueue,
			kernel,
			1,
			NULL,
			globalWorkSize,
			localWorkSize,
			0,
			NULL,
			&timing_event); CHECK(status);

	clFinish(cmdQueue);

	clGetEventProfilingInfo(timing_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &starttime,
                                NULL);

        clGetEventProfilingInfo(timing_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endtime,
                                NULL);

        kernel_elapsed = (unsigned long)(endtime - starttime);
        fprintf(stdout, "1, Kernel Execution\t%f sec\n", (double)(kernel_elapsed/1000000000.0));
	status = clEnqueueBarrier(cmdQueue); CHECK(status);

	// if the number of elements is bigger than BUFFERSIZE we need to
	// use additional sorting stages
	for (i = (unsigned int)(log2(BUFFERSIZE)) + 1; i <= iterations; i++)
	{
		// enqueue a barrier in the cmd queue so that every new
		// instance will start only when the previous one has been
		// completed

		// Reset the kernel arguments for the new instance
		status = clSetKernelArg(
				kernel,
				0,
				sizeof(cl_mem),
				(void*)&bufferX); CHECK(status);

		status |= clSetKernelArg(
				kernel,
				1,
				sizeof(cl_mem),
				(void*)&bufferY); CHECK(status);

		status |= clSetKernelArg(
				kernel,
				2,
				sizeof(unsigned int),
				(void*)&output_size); CHECK(status);

		status |= clSetKernelArg(
				kernel,
				3,
				sizeof(unsigned int),
				(void*)&i); CHECK(status);

		// set the number of total threads to the number
		// of sub buffers that have to be sorted at the
		// iteration i - example: elements = 1024, i = 10,
		// globalWorkSize = 1 thread. Indeed the first stage
		// generates 2 sorted array of 512 elements in size
		// and the stage 10 will generate one sorted array
		// of 1024 elements in size
		globalWorkSize[0] = elements/pow(2, i);
		localWorkSize[0] = 1;
		// Execute the kernel
		status = clEnqueueNDRangeKernel(
				cmdQueue,
				kernel,
				1,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
				&timing_event); CHECK(status);
		clFinish(cmdQueue);
		clGetEventProfilingInfo(timing_event,
					CL_PROFILING_COMMAND_START,
					sizeof(cl_ulong),
					&starttime,
					NULL);

		clGetEventProfilingInfo(timing_event,
					CL_PROFILING_COMMAND_END,
					sizeof(cl_ulong),
					&endtime,
					NULL);

		kernel_elapsed = (unsigned long)(endtime - starttime);
		fprintf(stdout, "%d, Kernel Execution\t%f sec\n", i, (double)(kernel_elapsed/1000000000.0));
		cumulative += kernel_elapsed;
		status = clEnqueueBarrier(cmdQueue); CHECK(status);
	}

	// Use clEnqueueReadBuffer() to read the OpenCL output
	if ( elements <= BUFFERSIZE )
	{
		// if elements <= BUFFERSIZE
		// the output will be on bufferY
		clEnqueueReadBuffer(
				cmdQueue,
				bufferY,
				CL_TRUE,
				0,
				datasize,
				Y,
				0,
				NULL,
				NULL); CHECK(status);
	}
	else
	{
		// if elements > BUFFERSIZE the output may be
		// either on bufferX or bufferY depending on the
		// number of the sorting stages required
		if ( (iterations % 2) == 0)
		{
			clEnqueueReadBuffer(
					cmdQueue,
					bufferX,
					CL_TRUE,
					0,
					datasize,
					Y,
					0,
					NULL,
					NULL); CHECK(status);
		}
		else
		{
			clEnqueueReadBuffer(
					cmdQueue,
					bufferY,
					CL_TRUE,
					0,
					datasize,
					Y,
					0,
					NULL,
					NULL); CHECK(status);
		}
	}

	// stop measuring the time
	(void)gettimeofday(&stop_time, NULL);
	cl_elapsed = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
		     (stop_time.tv_usec - start_time.tv_usec);

	// start measuring the time for the serial merge sort implementation
	(void)gettimeofday(&start_time, NULL);
	(void)merge_sort(K, elements);
	// stop measuring the time
	(void)gettimeofday(&stop_time, NULL);
	cpu_elapsed = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
		        (stop_time.tv_usec - start_time.tv_usec);

	// start measuring the time for the naive sort implementation
	(void)gettimeofday(&start_time, NULL);
//	(void)naive_sort(Z, elements);
	// stop measuring the time
	(void)gettimeofday(&stop_time, NULL);
	naive_elapsed = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
		        (stop_time.tv_usec - start_time.tv_usec);

	// check the result is correct
	i = 0;
	while ((Y[i++] == K[i++]) && (i < elements));

	if (i == elements)
	{
		fprintf(stderr, "\nthe arrays match!\n\n");
	}
	else
	{
		fprintf(stderr, "\nthe arrays do not match!\n\n");
		for (i = 0; i < elements; i++)
		{
			fprintf(stderr, "%u ", Y[i]);
		}
		fprintf(stderr, "\n");
	}

	// print timings on the screen

	fprintf(f_out, "Throughput = %f MElements/s\n\n",
					(double)(1.0e-6 * elements/(double)(cl_elapsed/1000000.0)));
	fprintf(f_out, "buf_size,cl_elapsed,cpu_elapsed,naive_elapsed\n");
	fprintf(f_out, "%u,%f,%f,%f\n", (unsigned int) elements,
					(double)(cl_elapsed/1000000.0),
				     	(double)(cpu_elapsed/1000000.0),
				     	(double)(naive_elapsed/1000000.0));
//----------------------------------------------
// STEP 13: Release the OpenCL resources
//----------------------------------------------

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferX);
	clReleaseMemObject(bufferY);
	clReleaseContext(context);

	// Free host resources
	free(platforms);
	free(devices);
	free(X);
	free(Y);
}
