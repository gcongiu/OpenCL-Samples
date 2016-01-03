// System includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// OpenCL includes
#include "CL/opencl.h"

#define CHECK(status) 								\
	if (status != CL_SUCCESS)						\
	{									\
		fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
		exit(1);							\
	}									\

void naive_sort(unsigned int * vector, size_t size)
{
	unsigned int i, j, tmp;

	for (i = 0; i < size; i++)
	{
		for (j = i; j < size; j++)
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

// merge sort function
void merge_sort(unsigned int* vector, size_t size)
{
	// define indexes
	unsigned int k, i, j;

	// sorted and unsorted vector pointers
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

	FILE * f_out = stderr;

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
	cl_ulong starttime, endtime;
	unsigned long kernel_elapsed;

	// This code executes on the OpenCL host

	// Host data
	unsigned int *X = NULL; // Input/Output array to/from accelerator
	unsigned int *Y = NULL; // Output array from the accelerator
	unsigned int *Z = NULL; // naive sort
	unsigned int *K = NULL; // CPU merge sort

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

	cl_mem bufferX; // Input array on the device
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

	FILE * fp = fopen("mergesort.ptx", "rb");
	fseek(fp, 0, SEEK_END);
	size_t binarysize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	unsigned char * binary = (unsigned char *) malloc (binarysize);

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
	const char *kernel_name = "_sort";

	// Use clCreateKernel() to create a kernel
	kernel = clCreateKernel(program, (const char*)kernel_name, &status); CHECK(status);

	// number of sorting stages
	unsigned int iterations = (unsigned int)log2(elements);
	unsigned int output_size;

	(void)gettimeofday(&start_time, NULL);
	for (i = 1; i <= iterations; i++)
	{
//----------------------------------------------
// STEP 9: Set the kernel arguments
//----------------------------------------------

		output_size = (unsigned int)pow(2,i);
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

		// Define an index space (global work size) of work
		// items for execution. A workgroup size (local work
		// size) is not required, but can be used.
		size_t globalWorkSize[1];
		// There are 'elements' work-items
		globalWorkSize[0] = elements/pow(2, i);

//----------------------------------------------
// STEP 11: Enqueue the kernel for execution
//----------------------------------------------

		// Execute the kernel by using
		// clEnqueueNDRangeKernel().
		// 'globalWorkSize' is the 1D dimension of the
		// work-items
		status = clEnqueueNDRangeKernel(
				cmdQueue,
				kernel,
				1,
				NULL,
				globalWorkSize,
				NULL,
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
		status = clEnqueueBarrier(cmdQueue); CHECK(status);

	}

//----------------------------------------------
// STEP 12: Enqueue the kernel for execution
//----------------------------------------------

	// Use clEnqueueReadBuffer() to read the OpenCL output
	// buffer (bufferY)
	// to the host output array (Y)
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

	(void)gettimeofday(&stop_time, NULL);
	cl_elapsed = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
		     (stop_time.tv_usec - start_time.tv_usec);

//	fprintf(f_out, "cl_elapsed,cpu_elapsed,naive_elapsed\n");
//	fprintf(f_out, "%f,", (double)(cl_elapsed/1000000.0));

	(void)gettimeofday(&start_time, NULL);
	(void)merge_sort(K, elements);
	(void)gettimeofday(&stop_time, NULL);
	cpu_elapsed = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
		        (stop_time.tv_usec - start_time.tv_usec);

//	fprintf(f_out, "%f,", (double)(cpu_elapsed/1000000.0));

	(void)gettimeofday(&start_time, NULL);
//	(void)naive_sort(Z, elements);
	(void)gettimeofday(&stop_time, NULL);
	naive_elapsed = (stop_time.tv_sec - start_time.tv_sec) * 1000000 +
		        (stop_time.tv_usec - start_time.tv_usec);

//	fprintf(f_out, "%f\n", (double)(naive_elapsed/1000000.0));
#if 1
	// check sorted array
	i = 0;
	while ((Y[i++] == K[i++]) && (i < elements));

	if (i == elements)
		fprintf(f_out, "\nthe arrays match!\n\n");
	else
	{
		fprintf(f_out, "\nthe arrays does not match!\n\n");
		for (i = 0; i < elements; i++)
		{
			fprintf(stderr, "%u ", Y[i]);
		}
	}
#endif
	fprintf(stdout, "Throughput = %f MElements/s\n\n",
					(double)(1.0e-6 * elements/(double)(cl_elapsed/1000000.0)));
	fprintf(stdout, "buf_size,cl_elapsed,cpu_elapsed,naive_elapsede\n");
	fprintf(stdout, "%u,%f,%f,%f\n", elements,
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
