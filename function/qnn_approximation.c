/*****************************************************************

Qubit neural network
ex. Function Approximation Feed Forward network, Nov. 7 2017
N.Kouda, Nobuyuki Matsui 

compile:  gcc qnn_approximation.c -lm

execute: ./a.out [train file name] [result file name] [test file name] [test result file name]
ex) ./a.out func.dat res.dat test.dat testrslt.dat

This source code is released under the MIT License.



MIT License

Copyright (c) 2017 Noriaki Kouda, Nobuyuki Matsui

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

******************************************************************/


#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define INPUT_NUM 	2     	// num. of units (Input Layer)
#define HA_NUM 		14    	// num. of units (Hidden A Layer)
#define OUTPUT_NUM 	1     	// num. of units (Output Layer)

#define NUMTRAINPAT 21    	// Num. of patterns of train data
#define NUMTESTPAT 	21    	// MAXIMUM num. of patterns of test data

#define RSEED	66        	// Seed of random numbers
#define LR    0.1			// Learning rate

#define ITERATION  10000  	// MAXIMUM num. of learning iteration
#define Err_low	0.01      	// Error value for accomplishing trainning  
#define HENKAN (M_PI*0.5) 	// Convergence coefficient from value[0,1] to phase[0, \pi/2]
//#define HENKAN (M_PI)		// In function approximation training, convergence coefficient set \pi,
							// training may finishes in less learning iteration.

#define urand1  ( (double)(rand()%360)/180.0*M_PI - M_PI  ) // Generate random value [-\pi, \pi]


double net_in[NUMTRAINPAT][INPUT_NUM]; 		// input data to network 
double net_out[NUMTRAINPAT][OUTPUT_NUM];	// output of network
double tsignal[NUMTRAINPAT][OUTPUT_NUM];	// target signal 

double intoha[INPUT_NUM][HA_NUM];		// input of hidden layer
double sha[INPUT_NUM][HA_NUM];			// phase rotation \theta
double rha[HA_NUM];						// as threshold \lambda
double dha[HA_NUM];						// inverted degree \delta

double hatoout[HA_NUM][OUTPUT_NUM];		// input of output layer
double sout[HA_NUM][OUTPUT_NUM];		// phase rotation \theta
double rout[OUTPUT_NUM];				// as threshold \lambda
double dout[OUTPUT_NUM];				// inverted degree \delta
double output[OUTPUT_NUM];				// phase output of output layer


double dsha[INPUT_NUM][HA_NUM];			 
double drha[HA_NUM];
double ddha[HA_NUM];

double dsouttmp[HA_NUM][OUTPUT_NUM];
double dsout[HA_NUM][OUTPUT_NUM];
double drout[OUTPUT_NUM];
double ddout[OUTPUT_NUM];

double hasum_re[HA_NUM],hasum_im[HA_NUM];
double outsum_re[OUTPUT_NUM],outsum_im[OUTPUT_NUM];


void forward_propagation(int p);
void back_propagation(int p);
void dparam_reset(void);
void param_revision(void);
void read_file(FILE *fp, int patternnum);
void initialize(void);
double sigmoid(double);

int i;			// learning iteration
double lr;		// learning rate

int main(int argc,char **argv)
{
  char train_filename[100];
  char result_filename[100];
  char test_filename[100];
  char testresult_filename[100];
  char trainlog_filename[100];

  double verror;	// squared error value
  double dummy;		// temporary variable for calculation

  FILE *fp, *fp2;

  if(argc==1) {   /* print usage */
    fprintf(stderr,"usage : %s <training data> <result data> <test data> <test result data>\n",argv[0]);
    exit(1);
  }

  strcpy(train_filename,argv[1]);
  strcpy(result_filename,argv[2]);
  strcpy(test_filename,argv[3]);
  strcpy(testresult_filename,argv[4]);


  if ( train_filename[0]=='\0' ) { 
    fprintf(stderr,"specify training data file name\n");
    exit(1);
  }
  if ( result_filename[0]=='\0' ) {
    fprintf(stderr,"specify result data file name\n");
    exit(1);
  }
  if ( test_filename[0]=='\0' ) {  
    fprintf(stderr,"specify test data file name\n");
    exit(1);
  }
  if ( testresult_filename[0]=='\0' ) {
    fprintf(stderr,"specify test result data file name\n");
    exit(1);
  }

  srand(RSEED);    /* initialize */

  strcpy(trainlog_filename, train_filename);
  strcat(trainlog_filename, ".log");
  if((fp2=fopen(trainlog_filename,"w"))==NULL){
	fprintf(stderr,"Can't create ***File`%s` !\n",trainlog_filename);
	exit(1);
  }


  lr = LR;

  initialize();

  i = 0;
  while(1)
  { 
	 // Read training data 
	if((fp=fopen(train_filename,"r"))==NULL)
	{	
		fprintf(stderr,"***File`%s`open error !\n",train_filename);
		exit(1);
	}
	read_file(fp, NUMTRAINPAT);
	fclose(fp);

	dparam_reset();
	verror=0.0;
	
	// Train all patterns
	for(int j=0;j<NUMTRAINPAT;j++)
    {   
		forward_propagation(j);
		back_propagation(j);

		for(int k=0;k<OUTPUT_NUM;k++)
        {
			dummy=tsignal[j][k]-net_out[j][k];
			verror+=dummy*dummy;
		}
	}
	verror *= 0.5;
		
	//scanf("%d",&ch);    /*for debug*/
	printf("iter:%f	%d %lf\n",lr,i,verror);
	fprintf(fp2,"\%f %d	%lf\n",lr,i,verror);

	// Escape from loop if training is completed
	if(verror < Err_low || i >= ITERATION)
		break;

	param_revision();
	i++;

  }
  fclose(fp2);

  printf("\n Approximation result:\n");

  // Output of network approximating function in training data  
  printf("\ttrain data:\n");
  if((fp=fopen(result_filename,"w"))==NULL) { 
	fprintf(stderr,"***Can't create File`%s` !\n",result_filename);
	exit(1);
  }

  
  for(int j=0;j<NUMTRAINPAT;j++) {
	forward_propagation(j);

	for(int k=0;k<INPUT_NUM;k++){
		printf("%lf ",net_in[j][k]);		// input data
		fprintf(fp,"%lf ",net_in[j][k]);	
	}
		
	for(int k=0;k<OUTPUT_NUM;k++){
		printf("%lf ",net_out[j][k]);		// network output data
		fprintf(fp,"%lf ",net_out[j][k]);	
	}
		
	for(int k=0;k<(OUTPUT_NUM-1);k++){
		printf("(%lf) ",tsignal[j][k]);		// target data
		fprintf(fp,"(%lf) ",tsignal[j][k]);	
	}

	printf("(%lf)\n",tsignal[j][OUTPUT_NUM-1]);		// last target data
	fprintf(fp,"(%lf)\n",tsignal[j][OUTPUT_NUM-1]);	
  }

  fclose(fp);

  // Output of network approximating function in test data not trained  
  printf("\ttest data:\n");
  
  if((fp=fopen(test_filename,"r"))==NULL)
  {	 //Read test data
	fprintf(stderr,"***File`%s`open error !\n",test_filename);
	exit(1);
  }
  read_file(fp, NUMTESTPAT);
  fclose(fp);

  if((fp=fopen(testresult_filename,"w"))==NULL) {   // write test result
	fprintf(stderr,"***Can't create File`%s` !\n",testresult_filename);
	exit(1);
  }

  for(int j=0;j<NUMTESTPAT;j++) {
	forward_propagation(j);
	
	for(int k=0;k<INPUT_NUM;k++){
		printf("%lf ",net_in[j][k]);		// input data
		fprintf(fp,"%lf ",net_in[j][k]);	
	}
		
	for(int k=0;k<OUTPUT_NUM;k++){
		printf("%lf ",net_out[j][k]);		// network output data
		fprintf(fp,"%lf ",net_out[j][k]);	
	}
		
	for(int k=0;k<(OUTPUT_NUM-1);k++){
		printf("(%lf) ",tsignal[j][k]);		// target data
		fprintf(fp,"(%lf) ",tsignal[j][k]);	
	}

	printf("(%lf)\n",tsignal[j][OUTPUT_NUM-1]);		// last target data
	fprintf(fp,"(%lf)\n",tsignal[j][OUTPUT_NUM-1]);	
  }
  fclose(fp);

  return 0;
}


// Calc. forward propagation at p-th pattern
// p means p-th training pattern
void forward_propagation(int p)
{
  	double tmp;

    for(int j=0;j<INPUT_NUM;j++)
        for(int i=0;i<HA_NUM;i++)
    		intoha[j][i] = HENKAN*net_in[p][j];

  	for(int j=0;j<HA_NUM;j++)
  	{
    	hasum_re[j] = 0.0;
    	hasum_im[j] = 0.0;

    	for(int i=0;i<INPUT_NUM;i++)
    	{
      		hasum_re[j] += cos(intoha[i][j]+sha[i][j]);
      		hasum_im[j] += sin(intoha[i][j]+sha[i][j]);
    	}

    	hasum_re[j] -= cos(rha[j]);
    	hasum_im[j] -= sin(rha[j]);

    	tmp = atan2(hasum_im[j],hasum_re[j]);
    	tmp = (M_PI/2.0)*sigmoid(dha[j])-tmp;

    	for(int i=0;i<OUTPUT_NUM;i++)
    		hatoout[j][i] = tmp;
   	}

  	for(int j=0;j<OUTPUT_NUM;j++)
  	{
    	outsum_re[j] = 0.0;
    	outsum_im[j] = 0.0;

    	for(int i=0;i<HA_NUM;i++)
    	{
      		outsum_re[j] += cos(hatoout[i][j]+sout[i][j]);
      		outsum_im[j] += sin(hatoout[i][j]+sout[i][j]);
    	}

    	outsum_re[j] -= cos(rout[j]);
    	outsum_im[j] -= sin(rout[j]);

    	tmp = atan2(outsum_im[j],outsum_re[j]);
    	output[j] = (M_PI/2.0)*sigmoid(dout[j])-tmp;

        net_out[p][j] = sin(output[j]) * sin(output[j]);

   	}

}

// Calc. back propagation at p-th pattern
// p means p-th training pattern
void back_propagation(int p)
{

  	double cos_tmp,sin_tmp,denom_tmp,gzai_tmp,tmp;

	//Output Layer back propergation
  	for(int j=0;j<OUTPUT_NUM;j++)
    {
    	cos_tmp = outsum_re[j];
    	sin_tmp = outsum_im[j];

        denom_tmp = cos_tmp*cos_tmp + sin_tmp*sin_tmp;

        // Calc. \theta back prop 
        for(int i=0;i<HA_NUM;i++)
        {
    		tmp = (-1.0)*sin(2.0*output[j])*(tsignal[p][j] - net_out[p][j]);
    		
        	tmp = (-1.0)*tmp*(cos(sout[i][j]+hatoout[i][j])*cos_tmp
            					+ sin(sout[i][j]+hatoout[i][j])*sin_tmp);
        	dsouttmp[i][j] = tmp/denom_tmp;

            dsout[i][j] += dsouttmp[i][j];
        }

        // Calc. \lambda back prop 
        tmp = (-1.0)*sin(2.0*output[j])*(tsignal[p][j]-net_out[p][j]);
        tmp = tmp*(cos(rout[j])*cos_tmp+sin(rout[j])*sin_tmp);
        tmp = tmp / denom_tmp;

        drout[j] += tmp;

        // Calc. \delta back prop 
        tmp = (-1.0)*sin(2.0*output[j])*(tsignal[p][j]-net_out[p][j]);
        tmp = tmp*(M_PI/2.0)*sigmoid(dout[j])*(1.0-sigmoid(dout[j]));

        ddout[j] += tmp;
    }

	//Hidden Layer back propergation
  	for(int j=0;j<HA_NUM;j++)
    {
        cos_tmp = 0.0; sin_tmp = 0.0;
    	gzai_tmp=0.0;

    	for(int i=0;i<OUTPUT_NUM;i++)
           	gzai_tmp += dsouttmp[j][i];

       	cos_tmp = hasum_re[j];
    	sin_tmp = hasum_im[j];

        denom_tmp = cos_tmp*cos_tmp + sin_tmp*sin_tmp;

        // Calc. \theta back prop 
        for(int i=0;i<INPUT_NUM;i++)
        {
           	tmp = gzai_tmp*(-1.0)*(cos(sha[i][j]+intoha[i][j])*cos_tmp
        								+ sin(sha[i][j]+intoha[i][j])*sin_tmp);
        	tmp = tmp/denom_tmp;

            dsha[i][j] += tmp;
        }

        // Calc. \lambda back prop 
        tmp = gzai_tmp*(cos(rha[j])*cos_tmp+sin(rha[j])*sin_tmp);
        tmp = tmp / denom_tmp;

        drha[j] += tmp;

         // Calc. \delta back prop 
		tmp = gzai_tmp*(M_PI/2.0)*sigmoid(dha[j])*(1.0-sigmoid(dha[j]));

        ddha[j] += tmp;
    }

}

// Revise neuron parameters by revision variables
void param_revision(void)
{
   	for(int j=0;j<HA_NUM;j++)
    {
    	for(int i=0;i<INPUT_NUM;i++)
      		sha[i][j] -= lr*dsha[i][j];

    	rha[j] -= lr*drha[j];
        dha[j] -= lr*ddha[j];
	}

    for(int j=0;j<OUTPUT_NUM;j++)
    {
    	for(int i=0;i<HA_NUM;i++)
     		sout[i][j] -= lr*dsout[i][j];

    	rout[j] -= lr*drout[j];
        dout[j] -= lr*ddout[j];
	}
}


// Clear all revision variables
void dparam_reset(void)
{
	for(int j=0;j<OUTPUT_NUM;j++)	//Output Layer
    {
        for(int i=0;i<HA_NUM;i++)
            dsout[i][j] = 0.0;

        drout[j] = 0.0;
        ddout[j] = 0.0;
    }

  	for(int j=0;j<HA_NUM;j++)		//HA Layer
    {
        for(int i=0;i<INPUT_NUM;i++)
            dsha[i][j] = 0.0;

        drha[j] = 0.0;
        ddha[j] = 0.0;
    }
}

// Give initial values to neuron parameters
void initialize(void)
{
	for(int j=0;j<HA_NUM;j++)
    {
		for(int i=0;i<INPUT_NUM;i++)
      		sha[i][j]= urand1;

        rha[j] = urand1;
        dha[j] = urand1;
  	}

	for(int j=0;j<OUTPUT_NUM;j++)
    {
		for(int i=0;i<HA_NUM;i++)
      		sout[i][j]= urand1;

        rout[j] = urand1;
        dout[j] = urand1;
  	}
}


double sigmoid(double input)
{
  	double  output;

    output = 1.0/(1.0+exp((-1.0)*input));

    return output;
}


void read_file(FILE *fp, int patternnum)
{
  int i,j;

  for(i=0;i<patternnum;i++) {

    for(j=0;j<INPUT_NUM;j++)
      fscanf(fp,"%lf",&net_in[i][j]);

    for(j=0;j<OUTPUT_NUM;j++)
      fscanf(fp,"%lf",&tsignal[i][j]);

  }
}
