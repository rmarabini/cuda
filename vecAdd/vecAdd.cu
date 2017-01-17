/**
 * vecAdd: C = A + B.
 *  
 * Partially based on CUDA samples from CUDA 7.5 Toolkit 
 *
 */

#include <stdio.h>
#include <time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 *
 * Error checking is not performed for simplicity
 * 
*/
__global__ void
vecAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    cudaError_t error;

    // Use device 0 or 1    
    cudaSetDevice(0);

    int numElements = 1000000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host vectors A, B and C
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize the host input vectors
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device vectors A, B and C
    cudaEvent_t startMem, stopMem; 

    cudaEventCreate(&startMem);
    cudaEventCreate(&stopMem);
    cudaEventRecord(startMem, 0) ;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Create start and stop CUDA events to measure time
    cudaEvent_t start, stop;
    float time;

    // Copy the host input vectors A and B to the device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stopMem, 0) ;
    cudaEventElapsedTime(&time, startMem, stopMem) 
    printf("time=%f,  Host to Device bandwidth (GB/s): %fn", time, numElements * 1e-6 / time);
    
    // Launch the Vector Add CUDA Kernel
///*
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
    
      int threadsPerBlock = 1024;
      int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
      printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
      vecAdd<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, numElements);
      error=cudaGetLastError();
      if (error!=cudaSuccess)
         printf("Maaaaal!!\n!");
//    vecAdd<<<1,numElements>>>(d_A, d_B, d_C, numElements);
    
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );

    // Copy the device result vector to the host memory.
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print kernel execution time
    printf("Kernel execution time %f\n", time);

    // Testing some values
    for (int i=0; i<10; i++)
	printf("%d -> A+B(host)=%f, A+B(GPU)=%f\n",i, h_A[i]+h_B[i], h_C[i]); 
    
    for (int i=1024; i<1034; i++)
	printf("%d -> A+B(host)=%f, A+B(GPU)=%f\n",i, h_A[i]+h_B[i], h_C[i]); 
    
    // Free device global memory (no error checking)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Destroy event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Reset the device and exit
    cudaDeviceReset();

    printf("Done\n");
    return 0;
}

