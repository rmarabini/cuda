/* File:     mat_add.cu
 * Purpose:  Implement matrix addition on a gpu using cuda
 *
 * Output:   Result of matrix addition.  
 *
 * Notes:
 * 1.  There are m blocks with n threads each.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"
 //#include "utils/cuPrintf.cu"

/*---------------------------------------------------------------------
 * Kernel:   Mat_add
 * Purpose:  Implement matrix addition
 * In args:  A, B, m, n
 * Out arg:  C
 */
__global__ void Mat_add_Vector(float matIn[], float vRef[], float matOut[], int numVec, int vecDim) {
    int threadCol = blockIdx.y * blockDim.x + threadIdx.x;
    int threadRow = blockIdx.x ;
    //if (blockIdx.y==0 && blockIdx.x==0)
       printf("col=%d, row=%dwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww\n");
    int indexOfMatrix = threadCol + threadRow * vecDim;

    if(threadCol < vecDim )
        {
        matOut[indexOfMatrix] = matIn[indexOfMatrix] + vRef[threadCol];
        }
}  /* Mat_add */


/*---------------------------------------------------------------------
 * Function:  Fill_matrix
 * Purpose:   Fill an m x n matrix with random values
 * In args:   m, n
 * Out arg:   A
 */
void Fill_matrix(float A[], int m, int n) {
   int i, j;
//numVec, dimVec
   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         A[i*n+j]=rand()/(float)RAND_MAX;
}  /* Read_matrix */


/*---------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print an m x n matrix to stdout
 * In args:   title, A, m, n
 */
void Print_matrix(const char title[], float A[], int numVec, int dimVec, int m, int n) {
   int i, j;
   //numVec, dimVec
   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%.2f ", A[i*dimVec+j]);
      printf("\n");
   }  
}  /* Print_matrix */

void checkError(cudaError_t error, const char function[])
{

        if(error != cudaSuccess)
        {
                printf("\"%s\" has a problem with error code %d and desc: %s\n", function, error, cudaGetErrorString(error));
                exit(-1);
        }
}

bool checkIfMatricesEqual(float * mat1, float * mat2, float matSize)
{
    int i = 0;
    for( ; i < matSize; i++)
       if(mat1[i] != mat2[i]){
           printf("values different for i: %d\n", i);
		   printf("mat1[i] = %d, mat2[i] = %d\n", mat1[i], mat2[i]);		   
		   return false;
	   }

    return true;
}

/* Host code */
int main(int argc, char* argv[]) {
   size_t numVec = 10;//mat size
   size_t dimVec = 10;

   // variables for threads per block, number of blocks.
   int threadsPerBlock = 1024;//, blocksInGrid = 0;

   //create cuda event variables
   cudaEvent_t hostStart, hostStop, deviceStart, deviceStop;
   float timeDifferenceOnHost, timeDifferenceOnDevice;
   //initialize cuda timing variables
   cudaEventCreate(&hostStart);
   cudaEventCreate(&hostStop);
   cudaEventCreate(&deviceStart);
   cudaEventCreate(&deviceStop);

   float *h_A, *h_B, *h_C, *h_C2;//PC
   float *d_A, *d_B, *d_C;//GPU
   size_t size, matrixSize;

   /* Get size of matrices */
   printf("dimVec = %d, numVec = %d\n", dimVec, numVec);
   matrixSize = numVec*dimVec;
   size = matrixSize*sizeof(float);
   size_t vecSize=dimVec*sizeof(float);

   h_A = (float*) malloc(size);
   h_B = (float*) malloc(vecSize);
   h_C = (float*) malloc(size);
   h_C2 = (float*) malloc(size);
   printf("size=%d vecSize=%d\n",size,vecSize);
   Fill_matrix(h_A, numVec, dimVec);
   Fill_matrix(h_B, 1, dimVec);

   Print_matrix("A =", h_A, numVec, dimVec, 4, 5);
   Print_matrix("B =", h_B, numVec, dimVec, 1, 5);

   printf("Adding matrices on CPU...\n");
   cudaEventRecord(hostStart, 0);
   for(int i = 0 ; i < numVec; i++)
       for(int j = 0 ; j < dimVec; j++){
           h_C2[i*dimVec+j] = h_A[i*dimVec+j] + h_B[j];
           //printf("i=%d, j=%d C2=%f a=%f b=%f i*dimVec+j=%d\n",i,j,h_C2[i*dimVec+j],h_A[i*dimVec+j],h_B[j],i*dimVec+j);
            }
   cudaEventRecord(hostStop, 0);
   cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);
   printf("Matrix addition over. Time taken on CPU: %5.5f\n",     
          timeDifferenceOnHost);


   /* Allocate matrices in device memory */
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, vecSize);
   cudaMalloc(&d_C, size);

   /* Copy matrices from host memory to device memory */
   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, vecSize, cudaMemcpyHostToDevice);

   //create a proper grid block using dim3

   /* Invoke kernel using m thread blocks, each of    */
   /* which contains n threads                        */
   dim3 block(threadsPerBlock);
   dim3 grid( numVec, ceil((float)dimVec/threadsPerBlock) );
   double blocky = ((double)dimVec)/threadsPerBlock;
   printf("dimVec=%d, threadsPerBlock=%d, ceil=%d %f\n", dimVec, threadsPerBlock, ceil(blocky), blocky);
   printf("dimVec=%d, threadsPerBlock=%d, ceil=%d %f\n", dimVec, threadsPerBlock, ceil(blocky), blocky);
   printf("dimVec=%d, threadsPerBlock=%d, ceil=%d %f\n", dimVec, threadsPerBlock, ceil(blocky), blocky);
//   printf("block=%d, grid.x=%d, grid.y=%d\n"          ,threadsPerBlock,numVec, ceil((float)dimVec/threadsPerBlock));
//   printf("kkkk=%d %d %d", ceil((float)dimVec/threadsPerBlock),dimVec,threadsPerBlock);
   cudaEventRecord(deviceStart, 0);
   //d_A -> inMatrix, d_B vRef, d_C outMat
//block=1024, grid.x=10, grid.y=1024
   Mat_add_Vector<<<grid, block>>>(d_A, d_B, d_C, numVec, dimVec);
//error=invalid configuration argumentvalues different for i: 0

   cudaError_t code=cudaGetLastError();
   if (code)
       printf("error=%s",cudaGetErrorString(code));
   else
       printf("code=%d",code);
   cudaDeviceSynchronize();  
   cudaEventRecord(deviceStop, 0);

   /* Wait for the kernel to complete */
   cudaThreadSynchronize();
   cudaEventElapsedTime(&timeDifferenceOnDevice, deviceStart, deviceStop);

   /* Copy result from device memory to host memory */
   checkError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Matrix C Copy from device to Host");
	
   if(checkIfMatricesEqual(h_C, h_C2, matrixSize))
      printf("Kernels correct!\n");
   else
      printf("Kernel logic wrong!\n");
	
   printf("Finished addition on GPU. Time taken: %5.5f\n", timeDifferenceOnDevice);   
   printf("Speedup: %5.5f\n", (float)timeDifferenceOnHost/timeDifferenceOnDevice);

   Print_matrix("The sum (CPU) is: ", h_C2, numVec, dimVec, 4, 5);
   Print_matrix("The sum (GPU) is: ", h_C, numVec, dimVec, 4, 5);

   /* Free device memory */
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   /* Free host memory */
   free(h_A);
   free(h_B);
   free(h_C);

   return 0;
}  /* main */
