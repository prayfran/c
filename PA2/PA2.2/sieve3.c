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
   long   ii, i, j,n,m, times,r;
   long   count;
   long *primes;
   long bs;
   /* Time */

   double time;

   if ( argc > 1 ) {
	N  = atoi(argv[1]);
    bs = atoi(argv[2]);
   }
   /* Start Timer */

   initialize_timer ();
   start_timer();

   size = (N+1)*sizeof(char);
   mark = (char *)malloc(size);

   for (i=2; i<=N; i=i+1){
     mark[i]=0;
   }

   curr=2;       /*first prime*/
   while (curr*curr<=N) {
   for (i=curr*curr; i<=sqrt(N); i++)
     if (i%curr == 0) mark[i]=1;
   while (mark[++curr]) ; /* do nothing */
   /* now index has the first unmarked number, so ... */
   }
   

    primes=(long *)malloc(sizeof(long) * sqrt(N)); 
    count = 0;
   for(i = 3; i <=sqrt(N); i+=2){
        if(mark[i] == 0) {
		    primes[count] = i;
        	++count;
        }

   }
   
   
	
   for(ii=sqrt(N); ii<N; ii+= bs){
		for(j=0; j<count; j++){
			if(primes[j] == 0){
			   j=count;	
			}	
			for(m=ii; m < ii+bs ;m++){
			  if(m % primes[j] == 0){
			  times = m;
			  break;
			  //printf("%ld\n", m);
			  }
			}
			if(times> N){
			times = N;	
			}
				for(i=times; i < ii+bs; i +=primes[j]){
				mark[i] = 1;	
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
        printf("%ld ", i);
        j++;
     }
   }
   printf("\n");


   printf("elapsed time = %lf (sec)\n", time);

   free(mark);
   return 0;
}





