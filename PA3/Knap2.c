//
// File name   : knap.c
// Author      : DaeGon
// Modified by : Sanjay and Wim 
// Date        : Sep 20
// Description : Dynamic Programming for the 0/1 knapsack
//               problem.  The entire table is saved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"
#include <math.h>

#define    MAX(x,y)   ((x)>(y) ? (x) : (y))
#define    table(i,j)    table[(i)*(C+1)+(j)]

long * get_last_row(int start, int end, int capacity, long *weights, long *profits ){
	
	long ssize = (capacity+1) * sizeof(long);
	long * second_to_last = (long *)malloc(ssize);
	long * last = (long *)malloc(ssize);
	for (long j=0; j<=capacity; j++) {
		
		second_to_last[j]= 0;
        last[j]=0;
	

     //printf("stl : %d ", second_to_last[j]);
	}

	for(long i=start;i<=end;i++){
		for ( long j=0 ; j <= capacity; j++ ) {
		 
         if(j<weights[i]){

           last[j] = second_to_last[j];
         }  
         else{
	 
		   last[j]=MAX(second_to_last[j],profits[i]+second_to_last[j-weights[i]]);
	 
		 }
		 
       }
		for(long l=0;l<=capacity;l++){
		second_to_last[l]=last[l];
		last[l]=0;	
		} 
	}
	
    
    return second_to_last;
   }
   
void solve_kp(int start, long end, long C, long *weights, long *profits, long *solu, long curr, long depth){
	//printf("start: %d ", start);
	//printf("end: %d ", end);
	//printf("C: %d \n ", C);
	if(C == 0)
	{
		return;
	}

	if (start==end)
	{
		if(weights[start]<=C)
		{
		//	if(start > 37){
		//		printf("start: %d", start);
		//		printf("weight{start}: %d ", weights[start]);
		//		printf("C: %d \n ", C);
		//	}
			solu[start]=1;
		//	printf("Adding: %d \n", start);	
		}

			return;
	}
	long *A1,*A2;
	long ssize = (C+1) * sizeof(long);
	long mid1,mid2, first_half=0, second_half;
	A1 = (long *)malloc(ssize);
	A2 = (long *)malloc(ssize);
	long highest = 0;
	mid2 = start+((end-start)/2);
 	#pragma omp parallel if (curr<depth)	
 	A1=get_last_row(start, mid2,C,weights,profits);
	#pragma omp parallel if (curr<depth)
	A2=get_last_row(mid2+1, end, C, weights, profits);

	
	for(long k=0;k<=C;k++){
		if((A1[k]+A2[C-k]) > highest){
		  highest = A1[k]+A2[C-k];
		  first_half = k;
		  second_half = C-k;	
		}  
	}
	free(A1);
	free(A2);
	#pragma omp parallel if(curr<depth)
	solve_kp(start, mid2, first_half, weights, profits, solu,curr+1, depth);
	#pragma omp parallel if(curr<depth)
	solve_kp(mid2+1, end, C-first_half, weights, profits, solu,curr+1, depth); 
	return;	
}
   
int main(int argc, char **argv) {

   FILE   *fp;
   long    N, C, depth, curr;                   // # of objects, capacity 
   long    *weights, *profits;     // weights and profits
   long *test;
   int    verbose,count;

   // Temp variables
   long    i, j,l, size, storesize;
   
   // Time
   double time;

   // Read input file (# of objects, capacity, (per line) weight and profit )
   if ( argc > 1 ) {
      fp = fopen(argv[1], "r"); 
      if ( fp == NULL) {
         printf("[ERROR] : Failed to read file named '%s'.\n", argv[1]);
         exit(1);
      }
   } else {
      printf("USAGE : %s [filename].\n", argv[0]);
      exit(1);
   }

   if (argc > 2) depth = atoi(argv[2]); 
   if (argc > 3) verbose = atoi(argv[3]); else verbose =0;
   fscanf(fp, "%d %d", &N, &C);
   printf("The number of objects is %d, and the capacity is %d.\n", N, C);

   
   size    = N * sizeof(long);
   storesize = (C+1) *sizeof(int);
   
   
   weights = (long *)malloc(size);
   profits = (long *)malloc(size);
   test= (long *)malloc(storesize);
   if ( weights == NULL || profits == NULL ) {
      printf("[ERROR] : Failed to allocate memory for weights/profits.\n");
      exit(1);
   }

   for ( i=0 ; i < N ; i++ ) {
      count = fscanf(fp, "%d %d", &(weights[i]), &(profits[i]));
	  
      if ( count != 2 ) {
         printf("[ERROR] : Input file is not well formatted.\n");
         exit(1);
      }
   }

   fclose(fp);

   // Solve for the optimal profit
   size = (C+1) * sizeof(long);

      long *table;
      size  = (C+1) * N * sizeof(long);
      table = (long *)malloc(size);
     // if ( table == NULL ) {
  //       printf("[ERROR] : Failed to allocate memory for the whole table.\n");
 //        exit(1);
  //    }
   
   
   initialize_timer ();
   start_timer();

   long *solu=(long *)malloc(size);
   
   for (long v =0; v <= N;v++){
	solu[v] = 0;
   } 

   solve_kp(0,N-1,C,weights,profits,solu,curr,depth);

   stop_timer();
   time = elapsed_time ();
   long HA=0;
   for (long m =0; m<N; m++){
	if(solu[m]==1){
	HA = HA+profits[m];	
	}
   }
   printf("The optimal profit is %d \nTime taken : %lf.\n",HA, time);

   if(verbose==1){
   printf("Solution vector is: ");
	
   for(long n =0; n<N; n++){
	if (verbose==1){
	printf("%d ",solu[n]);
	}   
   } 
      printf("\n");

	}

   

   


  
   // End of "Solve for the optimal profit"

   // Backtracking
  //    int c=C;
      
   //   for ( i=N-1 ; i > 0 ; i-- ) {
     //    if ( c-weights[i] < 0 ) {
	   //printf("i=%d: 0 \n",i);
       //     solution[i] = 0;
       //  } else {
      /*      if ( table(i-1,c) > table(i-1,c-weights[i]) + profits[i] ) {

	      //printf("i=%d: 0 \n",i);
               solution[i] = 0;
            } else {
	      //printf("i=%d: 1 \n",i);
               solution[i] = 1;
               c = c - weights[i];
            }
         }
      } 
      //wim: first row does not look back
      if(c<weights[0]){
        //printf("i=0: 1 \n");
	solution[0]=0;
      } else {
        //printf("i=0: 0 \n");
        solution[0]=1;
      }

      printf("The optimal profit is %d \nTime taken : %lf.\n", table(N-1,C), time);
     

      if (verbose==1) {

      printf("Solution vector is: ");
      for (i=0 ; i<N ; i++ ) {
         printf("%d ", solution[i]);
      }
      printf("\n");
      }

      if (verbose==2) {
	for (j=1; j<=C; j++){
	  printf ("%d\t", j);
	  for (i=0; i<N; i++)
	    printf ("%d ", table(i, j));
	  printf("\n");
	}
      }
*/
   return 0;
}
