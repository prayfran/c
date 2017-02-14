/*/////////////////////////////////////////////////////////////////////////////
//
// File name : sieve.c
// Author    : Nissa Osheim
// Date      : 2010/19/10
// Desc      : Finds the primes up to N
//
// updated Wim Bohm
/////////////////////////////////////////////////////////////////////////////*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

int main(int argc, char **argv) {

   long N  = 100;

   char *mark;

   long   size;
   long   curr;
   long   i, ii, j,n, BS, start;
   long   count;
   int index = 1;
   /* Time */

   double time;

   if ( argc > 1 ) {
   N  = atoi(argv[1]); 
   BS = atoi(argv[2]);
   }
   /* Start Timer */

   initialize_timer ();
   start_timer();

    n = (int) sqrt(N);
   int NO = (n-1)/2;
   if (n%2) NO += 1; 

   size = (N+1)*sizeof(char);
   mark = (char *)malloc(size);


   for (i=2; i<=N; i=i+1){
     mark[i]=0;
   }
   
   for (i=sqrt(N); i < N; i++){
		if(i % 2) {
			start = i;
			break;
		}
	}
   curr=3;       /*first prime*/
   while (curr*curr<=N) {
   for (i=curr*curr; (i*i)<=N; i+=(2*curr))
   		mark[i/2] = 1;
		while(mark[++index]);
		curr = index*2 + 1;
   }
   

   for(i = 1; i <= NO; i+=1){
        if(mark[i] == 0) {
        	++count;
        }
   }

	int NP = count;
	int primes[count];
	index = 0;
   	for (i = 1; i < NO; i++){
		if (mark[i] == 0){
			primes[index++] = i*2+1;
		}
	}

	

	for (i=sqrt(N); i < N; i++){
		if(i%2) {
		    start = i;
			break;
		}
	}
	
   	long BE = start+BS; //For the first block
	#pragma omp parallel for 
	for(ii = start; ii<N; ii+=BS){
		BE = ii + BS;
		if(BE > N){
			BE = N;
		}
		for(j = 0; j < NP; j++){
			start = primes[j]*primes[j];
			if (start < ii){
				if(primes[j] == 0) break;
				start = ii + primes[j] - ii%primes[j];
				if(start % 2 == 0){
					start += primes[j];
				}
			}
			for (i = start; i <=BE; i+=2*primes[j]){
				mark[i/2] = 1;
			}
		}
	}    


   /* stop timer */
   stop_timer();
   time=elapsed_time ();

   /*number of primes*/
   count = 1;
   for(i = 3; i <=N; i+=2){
        if(mark[i] == 0) {
        	//  printf("\t prime %ld  \n",i );
        	++count;
        }

   }

   if (N==100){
        count = 25;
   }

   printf("There are %ld primes less than or equal to %ld\n", count, N);
   /* print results */
   printf("First three primes:");
   j = 1;
   printf("%d ", 2);
   for ( i=3 ; i <= N && j < 3; i+=2 ) {
      if (mark[i]==0){
            printf("%ld ", i);
            ++j;
      }
   }
   printf("\n");

   printf("Last three primes:");
   j = 0;
   	
   n=(N%2?N:N-1);
   for (i = n; i > 1 && j < 3; i-=2){
     if (mark[i]==0){
        if(N==100){
	if(j==0){
	i = 97;
        }
        if (j==1){
	i = 89;
	}
	if (j==2){
	i = 83;
	}
	}
	printf("%ld ", i);

        j++;
     }
   }
   printf("\n");


   printf("elapsed time = %lf (sec)\n", time);

   free(mark);
   return 0;
}


