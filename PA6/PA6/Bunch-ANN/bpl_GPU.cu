/*---------------------------------------------------------------------------------------------------------------*/
/// bpl.c
/// For CSU CS475 Fall 2016
/// Instructor: Sanjay Rajopadhye
/// GTA: Swetha Varadarajan
/// Based on code Created by Paul Tero at Existor Ltd as part of a neural networks tutorial
/// Modified by Swetha Varadarajan
/// Created: 2016-11-16
/*---------------------------------------------------------------------------------------------------------------*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h> 

#include "timer.h"
#include "util.h"
#include "bunch_ann.h"
#include "bpl136810Kernel.h"
#include "bpl257911Kernel.h"


int main(int argc, char** argv) 
{

/*---------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------Command line parsing--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  Params cmdLineArgs;
  parseCmdLineArgs(&cmdLineArgs,argc,argv);

/*---------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------Variable Declaration------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  /*Array description and its size in the comments next to its declation*/

  double **inputs;//Given inputs = total number of samples(S)*number of inputs per sample(N) 
  double **outputs;//Expected outputs = total number of samples(S)*number of outputs per sample(P) 

  double **X;//Input for a given iteration = bunch size(I)*number of inputs per sample(N+1(bias))
  double **Y;//Output for a given iteration = bunch size(I)*number of outputs per sample(P)

  double **Wxh; //Weights in between input and hidden layer = (N+1)*M
  double **Why; //Weights in between input and hidden layer = (M+1)*P
  double **dWxh; //Error Weights in between input and hidden layer = (N+1)*M
  double **dWhy; //Error Weights in between input and hidden layer = (M+1)*P

  double **Zh; //Weighted sum for hidden layer=I*M
  double **H;  // Activation values = I*(M+1)
  double **Zy; //Weighted sum for output layer=I*P 
  double **E;  //Calculated Errors = I*P
  double **P1; //Oredicted output = I*P
  double **P;  // (exp(Zy)) = I*P
  double *sum; //(summation of the P[i]s) = I
  
  double learningrate = 0.0001; /*learning rate */
  long b = cmdLineArgs.sample_per_iter;
  
  long k2 = cmdLineArgs.sample_total/b ; /*number of full bunches */
  long k3 = cmdLineArgs.sample_total-(k2*b); /* size of the partial bunch */

/*---------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------Memory allocations--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
 
  inputs  = (double**)malloc(cmdLineArgs.sample_total * sizeof(double*));
  outputs = (double**)malloc(cmdLineArgs.sample_total * sizeof(double*));
  
  sum	  = (double*)malloc((b)*sizeof(double));

  for(long i = 0; i < cmdLineArgs.sample_total; ++i )
  {
	inputs[i] =(double*)malloc(cmdLineArgs.N * sizeof(double));
	outputs[i]=(double*)malloc(cmdLineArgs.P * sizeof(double));
  }

  Wxh     = (double**)malloc((cmdLineArgs.N+1) * sizeof(double*));
  Why	  = (double**)malloc((cmdLineArgs.M+1) * sizeof(double*));
  dWxh    = (double**)malloc((cmdLineArgs.N+1) * sizeof(double*));
  dWhy	  = (double**)malloc((cmdLineArgs.M+1) * sizeof(double*));

  for(long i = 0; i < cmdLineArgs.N+1; ++i )
  {
	Wxh[i] =(double*)malloc(cmdLineArgs.M * sizeof(double));	
	dWxh[i]=(double*)malloc(cmdLineArgs.M * sizeof(double));
  }

  for(long i = 0; i < cmdLineArgs.M+1; ++i )
  {
	Why[i] =(double*)malloc(cmdLineArgs.P * sizeof(double));
	dWhy[i]=(double*)malloc(cmdLineArgs.P * sizeof(double));
  }

  X	  = (double**)malloc(b*sizeof(double*));
  E	  = (double**)malloc(b*sizeof(double*));
  P	  = (double**)malloc(b*sizeof(double*));
  P1  = (double**)malloc(b*sizeof(double*));
  H	  = (double**)malloc(b*sizeof(double*));
  Zh  = (double**)malloc(b*sizeof(double*));
  Zy  = (double**)malloc(b*sizeof(double*));

  for(long i = 0; i < b; ++i )
  {
  X[i]	  = (double*)malloc((cmdLineArgs.N+1)*sizeof(double));
  E[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  P[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  P1[i]    = (double*)malloc(cmdLineArgs.P*sizeof(double));
  H[i]	  = (double*)malloc((cmdLineArgs.M+1)*sizeof(double));
  Zh[i]	  = (double*)malloc(cmdLineArgs.M*sizeof(double));
  Zy[i]	  = (double*)malloc(cmdLineArgs.P*sizeof(double));
  }

  if( inputs == NULL || outputs == NULL || X == NULL|| H == NULL || dWxh == NULL || dWhy == NULL 
      || Zh == NULL || Zy == NULL || Wxh == NULL || Why == NULL|| E == NULL || P == NULL
	  || P1 == NULL || sum == NULL)
  {
    printf( "Could not allocate memory\n" );
    exit(0);
  }
/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Initializations--------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/

  initializeW(Wxh,(cmdLineArgs.N+1),cmdLineArgs.M);
  initializeW(Why,(cmdLineArgs.M+1),cmdLineArgs.P);
  initializeI(inputs,cmdLineArgs.sample_total,cmdLineArgs.N);
  initializeO(outputs,cmdLineArgs.sample_total,cmdLineArgs.P);

/*---------------------------------------------------------------------------------------------------------------*/
/*------------------------------------------------Training-------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
    // Invoke kernel for warm up
  bpl1368210<<<dimGrid, dimBlock>>>(device_A, device_B, device_C);

  // Synchronize to make sure everyone is done in the warmup.
  cudaThreadSynchronize();
  
  
  initialize_timer();
  start_timer();
  for (long t=0; t<cmdLineArgs.iter; t++) //Time loop
  {
	 for (long s=0; s<k2; s++) //Bunch loop
	  { 	
		for(long i=0;i<b;i++)
		{
		X[i][0]=H[i][0]=1;//bias setting
		//required input/output are copied from inputs/outputs to X and Y
	 	memcpy (&X[i][1], inputs[(s*b)+i], cmdLineArgs.N*sizeof(double)); 
		}
		Y = &outputs[s*b]; 
		double** temp;
		/*Forward Phase*/
		bpl1361810<<<dimGrid, dimBlock>>>(Zh,X,Wxh,b,cmdLineArgs.N+1,cmdLineArgs.M, 1); //Zh=X*Wxh
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl257911<<<dimGrid, dimBlock>>>(H,Zh,temp,b,cmdLineArgs.M,1,0,0,2); //H=f1(Zh)
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl1361810<<<dimGrid, dimBlock>>>(Zy,H,Why,b,cmdLineArgs.M+1,cmdLineArgs.P,3); //Zy=H*Why	
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		func(P,Zy,b,cmdLineArgs.P,0); //P=fn(Zy)	
		reduction(P,sum,b,cmdLineArgs.P);  //summation of probabilities for each training sample
		prob(P,P1,sum,b,cmdLineArgs.P); //P1=fn(P,sum)	
		bpl257911<<<dimGrid, dimBlock>>>(E,P1,Y,b,cmdLineArgs.P,0,0,0,5);	//E=P1-Y
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		/*Backprpagation Phase*/ 		
		bpl1361810<<<dimGrid, dimBlock>>>(dWhy,H,E,cmdLineArgs.M+1,b,cmdLineArgs.P,6); //dWhy=H'*E ('->transpose)		
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl257911<<<dimGrid, dimBlock>>>(Why,dWhy,temp,cmdLineArgs.M+1,cmdLineArgs.P,0,0,learningrate,7); //Why=fn(dwhy)
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl1361810<<<dimGrid, dimBlock>>>(H,Why,E,b,cmdLineArgs.M+1,cmdLineArgs.P,8); //H=Why*E'		
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl257911<<<dimGrid, dimBlock>>>(Zh,H,temp,b,cmdLineArgs.M,0,0,0,9); //Zh=f1"(H) ("->gradient of f1)		
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl1361810<<<dimGrid, dimBlock>>>(dWxh,X,Zh,cmdLineArgs.N+1,b,cmdLineArgs.M,10);	//dWxh=X'Zh
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl257911<<<dimGrid, dimBlock>>>(Wxh,dWxh,temp,cmdLineArgs.N+1,cmdLineArgs.M,0,0,learningrate,11);//Wxh=fn(dWxh)
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
	}
	if(k3)
	{
		for(long i=0;i<k3;i++)
		{
		X[i][0]=H[i][0]=1;
	 	memcpy (&X[i][1], inputs[(k2*b)+i], cmdLineArgs.N*sizeof(double));
		}
		Y = &outputs[k2*b];

		/*Forward Phase*/
		bpl1361810<<<dimGrid, dimBlock>>>(Zh,X,Wxh,b,cmdLineArgs.N+1,cmdLineArgs.M, 1); //Zh=X*Wxh
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl257911<<<dimGrid, dimBlock>>>(H,Zh,temp,b,cmdLineArgs.M,1,0,0,2); //H=f1(Zh)
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl1361810<<<dimGrid, dimBlock>>>(Zy,H,Why,b,cmdLineArgs.M+1,cmdLineArgs.P,3); //Zy=H*Why	
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		func(P,Zy,b,cmdLineArgs.P,0); //P=fn(Zy)	
		reduction(P,sum,b,cmdLineArgs.P);  //summation of probabilities for each training sample
		prob(P,P1,sum,b,cmdLineArgs.P); //P1=fn(P,sum)	
		bpl257911<<<dimGrid, dimBlock>>>(E,P1,Y,b,cmdLineArgs.P,0,0,0,5);	//E=P1-Y
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		/*Backprpagation Phase*/ 		
		bpl1361810<<<dimGrid, dimBlock>>>(dWhy,H,E,cmdLineArgs.M+1,b,cmdLineArgs.P,6); //dWhy=H'*E ('->transpose)		
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl257911<<<dimGrid, dimBlock>>>(Why,dWhy,temp,cmdLineArgs.M+1,cmdLineArgs.P,0,0,learningrate,7); //Why=fn(dwhy)
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl1361810<<<dimGrid, dimBlock>>>(H,Why,E,b,cmdLineArgs.M+1,cmdLineArgs.P,8); //H=Why*E'		
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl257911<<<dimGrid, dimBlock>>>(Zh,H,temp,b,cmdLineArgs.M,0,0,0,9); //Zh=f1"(H) ("->gradient of f1)		
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl1361810<<<dimGrid, dimBlock>>>(dWxh,X,Zh,cmdLineArgs.N+1,b,cmdLineArgs.M,10);	//dWxh=X'Zh
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();
		bpl257911<<<dimGrid, dimBlock>>>(Wxh,dWxh,temp,cmdLineArgs.N+1,cmdLineArgs.M,0,0,learningrate,11);//Wxh=fn(dWxh)
		error = cudaGetLastError();
		if (error != cudaSuccess) Cleanup(false);
		cudaThreadSynchronize();

	}	
   }

  stop_timer();
  double time = elapsed_time();
  printf( "Time: %lf\n",time);
/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Print outputs----------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
   if(cmdLineArgs.V)
   {
	/*Need the following 2 statements for Testing*/
	displayMatrix ("input/hidden weights", Wxh, cmdLineArgs.N+1, cmdLineArgs.M);
	displayMatrix ("hidden/output weights", Why, cmdLineArgs.M+1, cmdLineArgs.P);
	/* Useful for analyzing the accuracy of prediction */
	/*if(k3)
	{	
		displayVector ("last input", &X[k3-1][1], cmdLineArgs.N);
		displayVector ("last output", Y[k3-1], cmdLineArgs.P);
		displayVector ("predicted output",P1[k3-1], cmdLineArgs.P);
	}
	else
	{
		displayVector ("last input", &X[b-1][1], cmdLineArgs.N);
		displayVector ("last output", Y[b-1], cmdLineArgs.P);
		displayVector ("predicted output",P1[b-1], cmdLineArgs.P);
	}*/
   }
/*---------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------Free Memory------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------*/
free(inputs);
free(outputs);
free(X);
free(Zh);
free(Zy);
free(H);
free(E);
free(P);
free(P1);
free(sum);
free(Wxh);
free(Why);
free(dWxh);
free(dWhy);
/*-------------------------------------------------------END-----------------------------------------------------*/
return 0;
}

