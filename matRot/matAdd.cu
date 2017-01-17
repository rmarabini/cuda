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
__global__ void rotMat(float matIn[], float vRef[], float matOut[], int numVec, int vecDim) {
    int threadCol = blockIdx.y * blockDim.x + threadIdx.x;
    int threadRow = blockIdx.x ;
    int indexOfMatrix = threadCol + threadRow * vecDim;

    if(threadCol < vecDim )
        matOut[indexOfMatrix] = matIn[indexOfMatrix] + vRef[threadCol];
}  /* Mat_add */


/*---------------------------------------------------------------------
 * Function:  Fill_matrix
 * Purpose:   Fill an m x n matrix with random values
 * In args:   m, n
 * Out arg:   A
 */
void Fill_matrix(float A[], int dimX, int dimY) {
   int i, j;
//numVec, dimVec
   for (i = 0; i < dimX; i++)
      for (j = 0; j < dimY; j++)
          if(i==j )//or (i+j)==(dimX+1))
            A[i*dimY+j]=1.0f;
          else
            A[i*dimY+j]=0.0f;
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
   size_t dimX = 10;//mat size
   size_t dimY = 10;
   float fX0=dimX/2., fY0=dimY/2.;
   int  iX0=dimX/2, iY0=dimY/2;

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

   float *h_A, *h_B, *h_B2;//PC
   float *d_A, *d_B;//GPU
   size_t size, matrixSize;

   /* Get size of matrices */

   matrixSize = dimX*dimY;
   size = matrixSize*sizeof(float);

   h_A = (float*) malloc(size);
   h_B = (float*) calloc(size,1);
   h_B2 = (float*) calloc(size,1);
   Fill_matrix(h_A, dimX, dimY);

   //init rot Matrix
   float rotMat[2][2];
 
   rotMat[0][0] = 0.f;
   rotMat[0][1] = -1.f;
   rotMat[1][0] = +1.f;
   rotMat[1][1] = 0.f;

   Print_matrix("A =", h_A, dimX, dimY, 10, 10);
   printf("rotMat=\n%.3f %.3f \n %.3f %.3f\n\n",rotMat[0][0],rotMat[0][1],rotMat[1][0],rotMat[1][1]);
   printf("Rotating matrices on CPU...\n");
   cudaEventRecord(hostStart, 0);
   float xOut,yOut;
   float xIn, yIn;
   int iIn, jIn;
   for(int i = 0 ; i < dimX; i++)
       for(int j = 0 ; j < dimY; j++){
           xOut = i - fX0;
           yOut = j - fY0;
           xIn = rotMat[0][0] * xOut + rotMat[0][1] * yOut;
           yIn = rotMat[1][0] * xOut + rotMat[1][1] * yOut;
           iIn = int(xIn + fX0);
           jIn = int(yIn + fY0);
           h_B2[i*dimY+j] = h_A[iIn*dimY+jIn];
           //printf("i=%d, j=%d C2=%f a=%f b=%f i*dimVec+j=%d\n",i,j,h_C2[i*dimVec+j],h_A[i*dimVec+j],h_B[j],i*dimVec+j);
            }
   cudaEventRecord(hostStop, 0);
   cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);
   printf("Matrix addition over. Time taken on CPU: %5.5f\n",     
          timeDifferenceOnHost);
   Print_matrix("B2(CPU) =", h_B2, dimX, dimY, 10, 10);

   /* Allocate matrices in device memory */
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, size);

   /* Copy matrices from host memory to device memory */
   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

   //create a proper grid block using dim3

   /* Invoke kernel using m thread blocks, each of    */
   /* which contains n threads                        */

   dim3 block(threadsPerBlock);
   dim3 grid( dimX, (dimY+threadsPerBlock-1)/threadsPerBlock );
   cudaEventRecord(deviceStart, 0);
   //d_A -> inMatrix, d_B vRef, d_C outMat
//block=1024, grid.x=10, grid.y=1024
   rotMat<<<grid, block>>>(d_A, d_B, dimX, dimY, rotMat);
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
   checkError(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost), "Matrix B Copy from device to Host");
	
   if(checkIfMatricesEqual(h_B, h_B2, matrixSize))
      printf("Kernels correct!\n");
   else
      printf("Kernel logic wrong!\n");
	
   printf("Finished addition on GPU. Time taken: %5.5f\n", timeDifferenceOnDevice);   
   printf("Speedup: %5.5f\n", (float)timeDifferenceOnHost/timeDifferenceOnDevice);

   Print_matrix("The sum (CPU) is: ", h_B2, dimX, dimY, 4, 5);
   Print_matrix("The sum (GPU) is: ", h_B, dimX, dimY, 4, 5);

   /* Free device memory */
   cudaFree(d_A);
   cudaFree(d_B);

   /* Free host memory */
   free(h_A);
   free(h_B);
   free(h_B2);

   return 0;
}  /* main */
