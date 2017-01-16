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
void Print_matrix(char *title, float A[], int m, int n) {
   int i, j;
   
   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%.1f ", A[i*n+j]);
      printf("\n");
   }  
}  /* Print_matrix */


/* Host code */
int main(int argc, char* argv[]) {
   size_t m = 1000;
   size_t n = 1000;

   float *h_A, *h_B, *h_C;//PC
   float *d_A, *d_B, *d_C;//GPU
   size_t size;

   /* Get size of matrices */
   printf("m = %d, n = %d\n", m, n);
   size = m*n*sizeof(float);

   h_A = (float*) malloc(size);
   h_B = (float*) malloc(size);
   h_C = (float*) malloc(size);
   
   Fill_matrix(h_A, m, n);
   Fill_matrix(h_B, m, n);
   
   const char Amsg="A =";
   const char Bmsg="B =";
   Print_matrix(Amsg, h_A, 4, 5);
   Print_matrix(Bmsg, h_B, 4, 5);

   /* Allocate matrices in device memory */
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, size);
   cudaMalloc(&d_C, size);

   /* Copy matrices from host memory to device memory */
   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

   /* Invoke kernel using m thread blocks, each of    */
   /* which contains n threads                        */
   Mat_add<<<m, n>>>(d_A, d_B, d_C, m, n);

   /* Wait for the kernel to complete */
   cudaThreadSynchronize();

   /* Copy result from device memory to host memory */
   cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

   Print_matrix("The sum is: ", h_C, 4, 5);

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
