#include "bpl136810Kernel.h"

#define BLOCK_SIZE = 32

__global__ void bpl1361810(double** X, double** Y, double** Z, long A, long B, long C, int number)
{
	long i,j,k;
	// matrix blocks
  float *Xsub, *Ysub, *Zsub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Xsub = &X.elements[X.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Xvalue = 0;
  
	for (i=0; i<A; i++) 
		for (j=0; j<B; j++)
			if(number == 8){
				X[i][j]=0;
				#pragma unroll
				for(k=0; k<BLOCK_SIZE; k++) 
					{
						Xvalue += Y[j][k] * Z[i][k];
					}
					__syncthreads();
					X[i][j] += Xvalue;
				}
			else if (number == 6 || number == 10 || number == 1 || number == 3){
			#pragma unroll		
			for(k=0; k<BLOCK_SIZE; k++) 
			{
					if(j==0) X[i][k]=0;
					if (number == 1 || number == 3){
						Xvalue += Y[i][j] * Z[j][k];
					}else{	
						Xvalue += Y[j][i] * Z[j][k];
					}
					__syncthreads();
					X[i][k]=Xvalue;
			}
		
		}
	}
    
}
