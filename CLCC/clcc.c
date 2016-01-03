/************************************/
/*                                  */
/*     OpenCL offline compiler      */
/*     Author: Giuseppe Congiu      */
/*                                  */
/************************************/
#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// log max size
// #define LOG_SIZE 1024*1024

// input filename
char inputname[256];
// output filename
char outputname[256];
// build options
char options[64][128];
// string for logging
// char logString[LOG_SIZE];
// default device type
cl_device_type type = CL_DEVICE_TYPE_DEFAULT;
// check input parameters
void check_parameters(int argc, char *argv[]);
// get error messages from the cl compiler
void cl_check_errors();

int main (int argc, char* argv[])
{
	/* source file pointer */
	FILE *p_out_file;
	FILE *p_in_file ;

	/* source code length */
	size_t szSourceCode;
	size_t szBinary;
	size_t size;

	/* source code string container */
	char *clSourceCode 	= NULL;
	unsigned char *clBinary = NULL;

	/* check input parameters */
	(void)check_parameters(argc, argv);

	/* opencl parameters */
	cl_platform_id clPlatform;
	cl_device_id   clDevice;
	cl_context     clContext;
	cl_program     clProgram;
	cl_int	       err;

	/* open source file for reading */
	p_in_file = fopen(inputname,"rb");
	if (p_in_file == NULL)
	{
		fprintf(stderr,"Error opening file %s\n", inputname);
		exit(EXIT_FAILURE);
	}

	/* get code size */
	fseek(p_in_file, 0, SEEK_END);		// point to the end of the stream
	szSourceCode = ftell(p_in_file);	// get the actual position
	fseek(p_in_file, 0, SEEK_SET);		// reset pointer to the start of the stream

#ifdef _ENABLE_DEBUG_
	fprintf(stdout,"Source Code size: %d\n",szSourceCode);
	fflush(stdout);
#endif

	/* allocate clSourceCode string */
	clSourceCode = (char *)malloc(szSourceCode + 1);
	if (clSourceCode == NULL)
	{
		fprintf(stderr,"Error allocating space for clSourceCode !!!");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	/* load source code into clSourceCode string */
	size = fread(clSourceCode, 1, szSourceCode, p_in_file);
	if (size != szSourceCode)
	{
		fclose(p_in_file);
		free(clSourceCode);
		fprintf(stderr,"Error reading source code from file %s !!!\n", inputname);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

#ifdef _ENABLE_DEBUG_
	fprintf(stdout,"Bytes readed from source: %d\n", size);
	fflush(stdout);
#endif

	/* end string character */
	clSourceCode[szSourceCode + 1] = '\0';

	/* init cl params */
	clGetPlatformIDs(1, &clPlatform, NULL);
	clGetDeviceIDs(clPlatform, type, 1, &clDevice, NULL);
	clContext  = clCreateContext(NULL, 1, &clDevice, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		fprintf(stderr,"Error %d creating context !!!\n", err);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	/* create opencl program with source */
	clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&clSourceCode, (const size_t *)&szSourceCode, &err);
	if (err != CL_SUCCESS)
	{
		fprintf(stderr,"Error %d creating program with source !!!\n", err);
		fflush(stderr);
		exit(EXIT_FAILURE);
        }

	/* build program */
//	char *ptr = (char *)options;
	char *ptr = "-Werror";
	err = clBuildProgram(clProgram, 1, &clDevice, ptr, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		fprintf(stderr, "Error Building Program !!!\n");
		fflush(stderr);
		/* get build information */
		size_t log_size;
		char* logString;
		clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		logString = (char *) malloc (log_size);
		clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, log_size, logString, NULL);
		logString[log_size] = '\0';
		fprintf(stdout, "%s\n", logString);
		fflush(stdout);
		exit(EXIT_FAILURE);
	}
//	fprintf(stdout, "Program Built Correctly !!!\n");
//	fflush(stdout);

#if 0 /* get build infos */
	cl_build_status status;
	clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
	if (status != CL_BUILD_SUCCESS)
	{
		fprintf(stderr, "error building program !!!\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
#endif

#if 0 /* get build log */
	size_t log_size;
	char* logString;
	clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	logString = (char *) malloc (log_size);
	err = clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, log_size, logString, NULL);
	if (err != CL_SUCCESS)
	{
		fprintf(stderr, "Error getting program build log  !!!\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
	logString[log_size] = '\0';
	fprintf(stdout, "%s\n", logString);
	fflush(stdout);
#endif

#if 0 /* get binary informations */
	cl_uint numdevices;
	clGetProgramInfo(clProgram, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numdevices, NULL);
	fprintf(stdout,"Number of devices: %u\n", numdevices);
	cl_device_id device;
	clGetProgramInfo(clProgram, CL_PROGRAM_DEVICES, sizeof(cl_device_id), &device, NULL);
#endif
	size_t bytesCopied;
	err = clGetProgramInfo(clProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &szBinary, NULL /* &bytesCopied */);
	if (err != CL_SUCCESS)
	{
		fprintf(stderr,"Error %d\n", err);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

#ifdef _ENABLE_DEBUG_
//	fprintf(stdout,"Bytes copied: %d\n",bytesCopied);
	fprintf(stdout,"Binary size: %d\n", szBinary);
	fflush(stdout);
#endif

	clBinary = (unsigned char *) malloc (szBinary);
	if (clBinary == NULL)
	{
		fprintf(stderr,"Error allocating space for clBinary !!!");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	err = clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, szBinary, (unsigned char*)&clBinary, &bytesCopied);
	if (err != CL_SUCCESS)
	{
		fprintf(stderr,"Error %d\n", err);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

#ifdef _ENABLE_DEBUG_
	fprintf(stdout,"Bytes copied to clBinary: %d\n", bytesCopied);
	fflush(stdout);
#endif
	/* open target file for writing */
	p_out_file = fopen(outputname,"wb");
	if (p_out_file == NULL)
	{
		fprintf(stderr,"Error opening file %s\n", inputname);
		exit(EXIT_FAILURE);
	}

	/* copy binary code to file */
	size = fwrite(clBinary, 1, szBinary, p_out_file);
#ifdef _ENABLE_DEBUG_
	fprintf(stdout, "binary size = %d, byte written = %d\n", szBinary, size);
	fflush(stdout);
#endif
	if (size != (szBinary) )
	{
		fclose(p_out_file);
		free(clBinary);
		fprintf(stderr, "Error writing binary code to file !!!\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

#ifdef _ENABLE_DEBUG_
	fprintf(stdout,"Bytes copied to file: %d\n", size);
	fprintf(stdout, "Build successfull completed !!!\n");
	fflush(stdout);
#endif
	clReleaseContext(clContext);
	clReleaseProgram(clProgram);

	free(clSourceCode);
	free(clBinary);

	/* close file */
	fclose(p_in_file);
	fclose(p_out_file);

	return 0;
}

void check_parameters(int argc, char *argv[])
{
	int i, j;

	if (argc < 3)
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "Usage: clcc -o <outputfile.ptx> -c <inputfile.cl>\n");
		fprintf(stderr, "\n");
		exit(EXIT_FAILURE);
	}

	for (i = 1; (i < argc) && ((argv[i])[0] == '-'); i++)
	{
		switch ((argv[i])[1])
		{
			case 'o':
				i++;
				strncpy(outputname, argv[i], 256);
				break;
			case 'c':
				i++;
				strncpy(inputname, argv[i], 256);
				break;
			case 'd':
				i++;
				if (!strcmp("cpu", argv[i]))
				{
					type = CL_DEVICE_TYPE_CPU;
					break;
				}
				if (!strcmp("gpu", argv[i]))
				{
					type = CL_DEVICE_TYPE_GPU;
					break;
				}
			case 'f':
				break;
			default:
				fprintf(stderr, "\n");
				fprintf(stderr, "Usage: clcc -o <outputfile.cl.o> -c <inputfile.cl> -f <compiling options>\n");
				fprintf(stderr, "\n");
				break;
		}
	}
}

void cl_check_errors()
{
}
