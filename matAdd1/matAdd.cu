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

/*---------------------------------------------------------------------
 * Kernel:   Mat_add
 * Purpose:  Implement matrix addition
 * In args:  A, B, m, n
 * Out arg:  C
 */
__global__ void Mat_add(float A[], float B[], float C[], int m, int n) {
   /* blockDim.x = threads_per_block                            */
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int my_ij = blockDim.x * blockIdx.x + threadIdx.x;

   /* The test shouldn't be necessary */
   if (blockIdx.x < m && threadIdx.x < n) 
      C[my_ij] = A[my_ij] + B[my_ij];
}  /* Mat_add */


/*---------------------------------------------------------------------
 * Function:  Fill_matrix
 * Purpose:   Fill an m x n matrix with random values
 * In args:   m, n
 * Out arg:   A
 */
void Fill_matrix(float A[], int m, int n) {
   int i, j;

   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         A[i*n+j]=rand()/(float)RAND_MAX;
}  /* Read_matrix */


/*---------------------------------------------------------------------
 * Function:  Print_matrix
 * Purpose:   Print an m x n matrix to stdout
 * In args:   title, A, m, n
 */
void Print_matrix(const char title[], float A[], int m, int n) {
   int i, j;
   
   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%.1f ", A[i*n+j]);
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
   size_t m = 1000;//mat size
   size_t n = 1000;

   // variables for threads per block, number of blocks.
   int threadsPerBlock = 16;//, blocksInGrid = 0;

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
   printf("m = %d, n = %d\n", m, n);
   matrixSize = m*n;
   size = matrixSize*sizeof(float);

   h_A = (float*) malloc(size);
   h_B = (float*) malloc(size);
   h_C = (float*) malloc(size);
   h_C2 = (float*) malloc(size);
   
   Fill_matrix(h_A, m, n);
   Fill_matrix(h_B, m, n);

   Print_matrix("A =", h_A, 4, 5);
   Print_matrix("B =", h_B, 4, 5);

   printf("Adding matrices on CPU...\n");
   cudaEventRecord(hostStart, 0);
   for(int i = 0 ; i < m*n; i++)
           h_C2[i] = h_A[i] + h_B[i];
   Print_matrix("CPU =", h_C2, 4, 5);
   cudaEventRecord(hostStop, 0);
   cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);
   printf("Matrix addition over. Time taken on CPU: %5.5f\n",     
          timeDifferenceOnHost);


   /* Allocate matrices in device memory */
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, size);
   cudaMalloc(&d_C, size);

   /* Copy matrices from host memory to device memory */
   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

   //create a proper grid block using dim3

   /* Invoke kernel using m thread blocks, each of    */
   /* which contains n threads                        */
   dim3 block(threadsPerBlock,threadsPerBlock);
   dim3 grid( (n + threadsPerBlock - 1/block.x), 
              (m + block.y - 1/block.y));

   cudaEventRecord(deviceStart, 0);
   Mat_add<<<block, grid>>>(d_A, d_B, d_C, m, n);
   cudaEventRecord(deviceStop, 0);

   /* Wait for the kernel to complete */
   cudaThreadSynchronize();
   cudaEventElapsedTime(&timeDifferenceOnDevice, deviceStart, deviceStop);

   /* Copy result from device memory to host memory */
   checkError(cudaMemcpy(d_C, h_C, size, cudaMemcpyDeviceToHost), "Matrix C Copy from device to Host");
	
   if(checkIfMatricesEqual(h_C, h_C2, matrixSize))
      printf("Kernels correct!\n");
   else
      printf("Kernel logic wrong!\n");
	
   printf("Finished addition on GPU. Time taken: %5.5f\n", timeDifferenceOnDevice);   
   printf("Speedup: %5.5f\n", (float)timeDifferenceOnHost/timeDifferenceOnDevice);

   Print_matrix("The sum (CPU) is: ", h_C2, 4, 5);
   Print_matrix("The sum (GPU) is: ", h_C, 4, 5);

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
