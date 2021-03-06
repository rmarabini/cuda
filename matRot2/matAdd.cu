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

__global__ void rotMatFunc(float matIn[], 
                           float matOut[], 
                           int dimX, 
                           int dimY, 
                           float rotMat[]) {
//    int y = blockIdx.y;
///    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x >= dimX || y >= dimY) 
         return;

    //// compute target address
    int  x0=dimX/2, y0=dimY/2;
    const unsigned int idx = x + y * dimX;

    const int xA = (x - x0 );
    const int yA = (y - y0 );

    const float xR = ( xA * rotMat[0] - yA * rotMat[1]);
    const float yR = ( -xA * rotMat[2] + yA * rotMat[3]);
    float src_x = xR + x0;
    float src_y = yR + y0;

     if ( src_x >= 0.0f && src_x < dimX && src_y >= 0.0f && src_y < dimY) {
        // BI - LINEAR INTERPOLATION
        float src_x0 = (float)(int)(src_x);
        float src_x1 = (src_x0+1); if(src_x1 == dimX) src_x1=src_x0;
        float src_y0 = (float)(int)(src_y);
        float src_y1 = (src_y0+1); if(src_y1 == dimY) src_y1=src_y0;

        float sx = (src_x-src_x0);
        float sy = (src_y-src_y0);


        int idx_src00 = min(src_x0   + src_y0 * dimX,dimX*dimY-1.0f);
        int idx_src10 = min(src_x1   + src_y0 * dimX,dimX*dimY-1.0f);
        int idx_src01 = min(src_x0   + src_y1 * dimX,dimX*dimY-1.0f);
        int idx_src11 = min(src_x1   + src_y1 * dimX,dimX*dimY-1.0f);

        matOut[idx]  = (1.0f-sx)*(1.0f-sy)*matIn[idx_src00];
        matOut[idx] += (     sx)*(1.0f-sy)*matIn[idx_src10];
        matOut[idx] += (1.0f-sx)*(     sy)*matIn[idx_src01];
        matOut[idx] += (     sx)*(     sy)*matIn[idx_src11];
    } else {
        matOut[idx] = 0.0f;
     }




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
   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%.1f ", A[i*dimVec+j]);
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
void rotateCPU(float matIn[], 
               float matOut[], int dimX, int dimY,
               float rotMat[])
{
   int  x0=dimX/2, y0=dimY/2;
   for(int x = 0 ; x < dimX; ++x)
       for(int y = 0 ; y < dimY; ++y){
    //// compute target address
    const unsigned int idx = x + y * dimX;

    const int xA = (x - x0 );
    const int yA = (y - y0 );

    const float xR = ( xA * rotMat[0] - yA * rotMat[1]);
    const float yR = ( -xA * rotMat[2] + yA * rotMat[3]);
    float src_x = xR + x0;
    float src_y = yR + y0;

     if ( src_x >= 0.0f && src_x < dimX && src_y >= 0.0f && src_y < dimY) {
        // BI - LINEAR INTERPOLATION
        float src_x0 = (float)(int)(src_x);
        float src_x1 = (src_x0+1); if(src_x1 == dimX) src_x1=src_x0;
        float src_y0 = (float)(int)(src_y);
        float src_y1 = (src_y0+1); if(src_y1 == dimY) src_y1=src_y0;

        float sx = (src_x-src_x0);
        float sy = (src_y-src_y0);


        int idx_src00 = min(max(0.0f,src_x0   + src_y0 * dimX),dimX*dimY-1.0f);
        int idx_src10 = min(max(0.0f,src_x1   + src_y0 * dimX),dimX*dimY-1.0f);
        int idx_src01 = min(max(0.0f,src_x0   + src_y1 * dimX),dimX*dimY-1.0f);
        int idx_src11 = min(max(0.0f,src_x1   + src_y1 * dimX),dimX*dimY-1.0f);

        matOut[idx]  = (1.0f-sx)*(1.0f-sy)*matIn[idx_src00];
        matOut[idx] += (     sx)*(1.0f-sy)*matIn[idx_src10];
        matOut[idx] += (1.0f-sx)*(     sy)*matIn[idx_src01];
        matOut[idx] += (     sx)*(     sy)*matIn[idx_src11];
    } else {
        matOut[idx] = 0.0f;
     }
  }//for
}
/* Host code */
int main(int argc, char* argv[]) {
   size_t dimX = 9;//mat size
   size_t dimY = 9;
   int numberRot;
   printf("number of steps: ");
   scanf ("%d",&numberRot); //dimY=dimX; 

   // variables for threads per block, number of blocks.
   int threadsPerBlockX = 32;//, blocksInGrid = 0;   
   int threadsPerBlockY = 32;//, blocksInGrid = 0;

   //threadsPerBlock = min(_dimY, _dimY);
   //create cuda event variables
   cudaEvent_t hostStart, hostStop, deviceStart, deviceStop;
   float timeDifferenceOnHost, timeDifferenceOnDevice;
   //initialize cuda timing variables
   cudaEventCreate(&hostStart);
   cudaEventCreate(&hostStop);
   cudaEventCreate(&deviceStart);
   cudaEventCreate(&deviceStop);

   float *h_A, *h_B, *h_B2, *h_rotMat;//PC
   float *d_A, *d_B, *d_rotMat;//GPU
   size_t size, matrixSize;

   /* Get size of matrices */

   matrixSize = dimX*dimY;
   size = matrixSize*sizeof(float);

   h_A = (float*) calloc(size,1);
   h_B = (float*) calloc(size,1);
   h_B2 = (float*) calloc(size,1);
   h_rotMat = (float*) calloc(4*sizeof(float),1);

   /* Allocate matrices in device memory */
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, size);
   cudaMalloc(&d_rotMat, 4*sizeof(float));

   Fill_matrix(h_A, dimX, dimY);
   float initAngle=0;
   float lastAngle=90;
   float stepSize= (lastAngle - initAngle )/ numberRot;
   Print_matrix("A =", h_A, dimX, dimY, 9, 9);
   for (float angle=initAngle; angle <=lastAngle; angle+=stepSize){
       //init rot Matrix
       h_rotMat[0] = cosf(angle);
       h_rotMat[1] = sinf(angle);
       h_rotMat[2] = -h_rotMat[1];
       h_rotMat[3] =  h_rotMat[0];

      printf("Rotating matrices on CPU...\n");
      cudaEventRecord(hostStart, 0);
      //rotate matrix using CPU
      memset(h_B2, 0, size);
      rotateCPU(h_A ,h_B2, dimX, dimY, h_rotMat);
      cudaEventRecord(hostStop, 0);
      cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);
      printf("Matrix rotation over. Time taken on CPU: %5.5f\n",     
          timeDifferenceOnHost);
      if (angle==initAngle)
          Print_matrix("B2(CPU) =", h_B2, dimX, dimY, 9, 9);

      /* Copy matrices from host memory to device memory */
      memset(h_B, 0, size);
      cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_rotMat, h_rotMat, 4*sizeof(float), cudaMemcpyHostToDevice);

      /* Invoke kernel using dimX * dimY thread blocks, each of    */
      /* which contains threadsPerBlock threads                        */
      dim3 block(threadsPerBlockX, threadsPerBlockY);   
      dim3 grid;
      grid.x = (dimX + block.x - 1)/block.x;
      grid.y = (dimY + block.y - 1)/block.y;
      cudaEventRecord(deviceStart, 0);
      rotMatFunc<<<grid, block>>>(d_A, d_B, dimX, dimY, d_rotMat);
      cudaError_t code=cudaGetLastError();
      if (code)
         printf("error=%s",cudaGetErrorString(code));
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
      printf("GPUtime: %5.5f\n", (float)timeDifferenceOnDevice);

      Print_matrix("The rotated image(CPU) is: ", h_B2, dimX, dimY, 9, 9);
      Print_matrix("The rotated image(GPU) is: ", h_B, dimX, dimY, 9, 9);
      }
   /* Free device memory */
   cudaFree(d_A);
   cudaFree(d_B);

   /* Free host memory */
   free(h_A);
   free(h_B);
   free(h_B2);

   return 0;
}  /* main */
