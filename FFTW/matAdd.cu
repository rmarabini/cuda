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
//#include <helper_cuda.h>         // helper functions for CUDA error check
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
//ecurso01@finlay:~/ROB/cuda/FFTW> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64
//curso01@finlay:~/ROB/cuda/FFTW> git pull ; /usr/local/cuda-7.5/bin/nvcc matAdd.cu -I /usr/local/scipion/software/include -L /usr/local/scipion/software/lib -lfftw3f -run -D__INTEL_COMPILER -lm -L /usr/local/cuda-7.5/lib64/ -lcuda -lcufft

#include "fftw3.h"
#include <complex.h>
#include <cufft.h>


/*---------------------------------------------------------------------
 * Kernel:   Mat_add
 * Purpose:  Implement matrix addition
 * In args:  A, B, m, n
 * Out arg:  C
 */


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

void Print_matrix_complex(const char title[], fftwf_complex A[], int dimY, int dimX, int m, int n) {
   int i, j;
   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++){
           //c=A[i*dimX+j];
         printf("%.2f%+.2fi ", A[i*dimX+j][0], A[i*dimX+j][1]);
               }
      printf("\n");
   }  
}  /* Print_matrix */

void Print_matrix_cu_complex(const char title[], cufftComplex A[], int dimY, int dimX, int m, int n) {
   int i, j;
   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++){
           //c=A[i*dimX+j];
         printf("%.2f%+.2fi ", A[i*dimX+j].x, A[i*dimX+j].y);
               }
      printf("\n");
   }       

   printf(">>> %d \n", sizeof(cufftComplex)); 
   for (i = 0; i < 9; i++) 
         printf("%.2f%+.2fi ", A[i].x, A[i].y);
}  /* Print_matrix */

void checkError(cudaError_t error, const char function[])
{

        if(error != cudaSuccess)
        {
                printf("\"%s\" has a problem with error code %d and desc: %s\n", function, error, cudaGetErrorString(error));
                exit(-1);
        }
}

bool checkIfMatricesEqual(fftwf_complex * mat1, fftwf_complex * mat2, float matSize)
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
void fftwCPU(float matIn[], 
               fftwf_complex matOut[], int dimX, int dimY){  
    fftwf_plan p2d;
    p2d = fftwf_plan_dft_r2c_2d(dimX, dimY, matIn,matOut,FFTW_ESTIMATE );
    fftwf_execute(p2d);
}

/* Host code */
int main(int argc, char* argv[]) {
   size_t dimX = 3;//mat size
   size_t dimY = 3;

   //create cuda event variables
   cudaEvent_t hostStart, hostStop, deviceStart, deviceStop;
   float timeDifferenceOnHost, timeDifferenceOnDevice;
   //initialize cuda timing variables
   cudaEventCreate(&hostStart);
   cudaEventCreate(&hostStop);
   cudaEventCreate(&deviceStart);
   cudaEventCreate(&deviceStop);

   float *h_A;//PC
   cufftComplex *d_B;//GPU
   cufftReal *d_A;
   size_t size, matrixSize;

   /* Get size of matrices */

   matrixSize = dimX*dimY;
   size = matrixSize*sizeof(float);
   int sizeFourier = dimY*(dimX/2+1)*sizeof(fftwf_complex);
   //typedef float cufftReal; is a single-precision, floating-point real data type. 
   h_A = (float*) calloc(size,1);
   cufftComplex  * h_B  =(cufftComplex *) malloc(sizeFourier);
   fftwf_complex * h_B2 =(fftwf_complex *) malloc(sizeFourier);

   /* Allocate matrices in device memory */
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, sizeFourier);

   Fill_matrix(h_A, dimX, dimY);

    h_A[0]=1.f; h_A[1]=2.f; h_A[2]=3.f; h_A[3]=9.f; 
    h_A[4]=8.f; h_A[5]=5.f; h_A[6]=1.f; h_A[7]=2.f; 
    h_A[8]=9.f; //h_A[9]=8.f; h_A[10]=7.f; h_A[11]=2.f;



   Print_matrix("original matrix is: ", h_A, dimX, dimY, 3, 3);
   printf("fftw on CPU...\n");
   cudaEventRecord(hostStart, 0);
   fftwCPU(h_A ,h_B2, dimX, dimY);
   cudaEventRecord(hostStop, 0);
   cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);
   Print_matrix_complex("The fft image(CPU) is: ", h_B2, dimY, dimX/2+1, 3, 2);
   printf("Matrix fft over. Time taken on CPU: %5.5f\n", timeDifferenceOnHost);

   //Create Plan
   cudaEventRecord(deviceStart, 0);
   cufftHandle plan;
   cufftPlan2d(&plan, dimX, dimY, CUFFT_R2C);
   //fprintf(stderr, "cufftPlan2d\n");

   /* Copy matrices from host memory to device memory */
   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, sizeFourier, cudaMemcpyHostToDevice);
   cufftExecR2C(plan, d_A, d_B);
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
/*
      if(checkIfMatricesEqual(h_B, h_B2, matrixSize))
          printf("Kernels correct!\n");
      else
         printf("Kernel logic wrong!\n");
*/	
      printf("Finished fft on GPU. Time taken: %5.5f\n", timeDifferenceOnDevice);   
      printf("Speedup: %5.5f\n", (float)timeDifferenceOnHost/timeDifferenceOnDevice);
      printf("GPUtime: %5.5f\n", (float)timeDifferenceOnDevice);

      Print_matrix_complex("The fft image(CPU) is: ", h_B2, dimY, dimX/2+1, 3, 2);
      Print_matrix_cu_complex("The fft image(GPU) is: ", h_B, dimY, dimX/2+1, 3, 2);
//Print_matrix_cu_complex("The fft image(GPU) is: ", h_B, dimX, dimY, 3, 3);
      
   /* Free device memory */
   cudaFree(d_A);
   cudaFree(d_B);

   /* Free host memory */
   free(h_A);
   free(h_B);
   free(h_B2);

   return 0;
}  /* main */
