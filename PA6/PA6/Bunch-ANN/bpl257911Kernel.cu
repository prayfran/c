///
/// vecAddKernel00.cu
/// For CSU CS575 Spring 2011
/// Instructor: Wim Bohm
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// This Kernel adds two Vectors A and B in C on GPU
/// without using coalesced memory access.
/// 

__global__ void bpl257911(double** X, double** Y,double** Z, long A, long B, long val, int N, double C,int number)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int threadStartIndex = blockStartIndex + (threadIdx.x);
    int EndIndex   = blockStartIndex + (blockDim.x*N);
	long i,j;
    for( i=threadStartIndex; i<A; i=i+(blockDim.x); ){
		for (j=threadStartIndex; j<B; j=j+(blockDim.x);){
			if (number==2){
				X[i][j+val] = foo(Y[i][j],val);
			}else if (number==5){
				X[i][j] = Y[i][j]-Z[i][j]; 
				
			}else if (number==9){
				X[i][j] = Y[i][j+1]*(1 - pow (tanh (X[i][j]), 2));
			}else if (number==7 || number==11){
				X(i,j) -= C*Y(i,j); 
			}
	
		}
	}
}
