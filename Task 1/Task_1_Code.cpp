
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <set>
#include <math.h>

/*** Set namespace ***/
using namespace std;

int main (int argc, char *argv[])
{

/*** Specify if graph data is directed or undirected ***/
int directedNodes=0;

/*** Specify maximum number of iterations and tolerance on convergence ***/
int maxIter=100;
double tol=1e-7;

/*** Specify input filename (must be in same directory) ***/
const char* fileName="facebook_combined.txt";

/*** Specify output filename ***/
const char* outFileName="Output_Task1.csv";

/*** Count number of links in file ***/
string line;
ifstream infile(fileName);
int number_of_lines  = 0;
while(getline(infile, line)){
    ++number_of_lines;
}
cout << "Number of links in file: " << number_of_lines << endl << endl;
int numLinks_int=number_of_lines;
double numLinks=number_of_lines;
if(directedNodes==0){
    numLinks=2*number_of_lines;
    numLinks_int=2*number_of_lines;
}

/*** Store in links and out links from file ***/
int inLink[numLinks_int];
int outLink[numLinks_int];
string line2;
ifstream myfile(fileName);

cout<<"Outlink  Inlink"<<endl;
for(int k=0; k<=number_of_lines-1; k++){
    getline(myfile, line2);
    istringstream iss(line2);
    iss >> outLink[k] >> inLink[k];
//    cout << outLink[k] << "\t"<< inLink[k] << endl;
}
cout << endl;

/*** Find lowest and highest nodes ***/
int minNode=min(*min_element(inLink,inLink+number_of_lines),*min_element(outLink,outLink+number_of_lines)) ;
int maxNode=max(*max_element(inLink,inLink+number_of_lines),*max_element(outLink,outLink+number_of_lines));
cout << "The lowest node is " << minNode << endl;
cout << "The highest node is "<< maxNode << endl << endl;
double numNodes=maxNode+1;

/*** Initialize matrix ***/
double** M = 0;
M = new double*[maxNode+1];
for(int m=0; m<=maxNode; m++){
    M[m] = new double[maxNode+1];
    for(int n=0; n<=maxNode; n++){
        M[m][n] = 0;
    }
}

/*** Insert one for each link ***/
for(int x=0; x<=number_of_lines-1; x++){
    M[inLink[x]][outLink[x]]+=1;
    if(directedNodes==0){
        M[outLink[x]][inLink[x]]+=1;
    }
}

/*** Calculate number of outlinks per node ***/
cout << "Number of out links per node:" << endl;
double numOutLinks[maxNode];
for(int y=0; y<=maxNode; y++){
    numOutLinks[y]=0;
    for(int z=0; z<=maxNode; z=z+1){
        numOutLinks[y]+=M[z][y];
    }
//    cout << numOutLinks[y] << endl;
}
cout << endl;

/*** Create stochastic matrix ***/
double d=0.85;
for(int y=0; y<=maxNode; y++){
    for(int z=0; z<=maxNode; z=z+1){
        M[z][y]=d*(M[z][y]/numOutLinks[y])+(1-d)/numNodes;
    }
}

/*** Spawn a parallel region explicitly scoping all variables ***/
int i, j, k, tid, nthreads, chunk;
double r[maxNode+1];
double r2[maxNode+1];
double sum_sq;
double normDiff=1;
chunk=1;

#pragma omp parallel shared(outLink,inLink,numOutLinks,M,r,r2,numLinks,nthreads,chunk,maxNode) private(tid,i,j,k)
{
/*** Find number of processors ***/
tid = omp_get_thread_num();
if (tid == 0){
    nthreads = omp_get_num_threads();
//    printf("Starting matrix vector multiply with %d threads\n",nthreads);
//    printf("Initializing matrices...\n");
}

/*** Initialize page rank array ***/
#pragma omp for schedule (static)
for(i=0; i<=maxNode; i++){
    r[i]=1/numNodes;
}

}

/*** =========================================================================== ***/
/*** BEGIN POWER ITERATIONS                                                      ***/
/*** =========================================================================== ***/
int iter;
for(iter=0; iter<=maxIter; iter++){
#pragma omp parallel shared(outLink,inLink,numOutLinks,M,r,r2,numLinks,nthreads,chunk,maxNode) private(tid,i,j,k)
{
/*** Initialize results array ***/
#pragma omp for schedule (static)
for (i=0; i<=maxNode; i++){
    r2[i]= 0;
}
/*** Do matrix multiply sharing iterations on outer loop ***/
/*** Display who does which iterations for demonstration purposes ***/
//printf("Thread %d starting matrix multiply...\n",tid);
#pragma omp for schedule (static)
for (i=0; i<=maxNode; i++){
//    printf("Thread=%d did row=%d\n",tid,i);
    for(j=0; j<=maxNode; j++){
        r2[i] += (M[i][j]*r[j]);
    }
}

/*** End parallel region ***/
}

/*** Check convergence criteria ***/
sum_sq=0;
for (i=0; i<=maxNode; i++){
    sum_sq += pow((r2[i]-r[i]),2);
}
normDiff=sqrt(sum_sq);
//
//cout << "Page rank n:" << endl;
//cout.precision(4);
//cout << r2[0] << endl;
//cout << r2[1] << endl;
//cout << r2[2] << endl << endl;

for (i=0; i<=maxNode; i++){
    r[i] = r2[i];
}

if(normDiff<=tol){
    cout << "Converged!" << endl;
    cout << "Number of iterations: " << iter << endl << endl;
    break;
}
if(iter==maxIter){
    cout << "Maximum number of iterations reached" << endl;
    cout << "Increase number of iterations or decrease tolerance" << endl << endl;
}
}
/*** =========================================================================== ***/
/*** END POWER ITERATIONS                                                        ***/
/*** =========================================================================== ***/
//
//cout << "Stochastic matrix:" << endl;
//cout.precision(4);
//cout << M[0][0] << "\t" << M[0][1] << "\t" << M[0][2] << endl;
//cout << M[1][0] << "\t" << M[1][1] << "\t" << M[1][2] << endl;
//cout << M[2][0] << "\t" << M[2][1] << "\t" << M[2][2] << endl << endl;

cout << "Page rank array:" << endl;
cout.precision(4);
cout << r[0] << endl;
cout << r[1] << endl;
cout << r[2] << endl << endl;

cout << "Page rank result:" << endl;
cout.precision(4);
cout << r2[0] << endl;
cout << r2[1] << endl;
cout << r2[2] << endl << endl;

cout<< normDiff << endl;

/*** Close Facebook data file ***/
infile.close();

/*** Write results to file ***/
ofstream outfile;
outfile.open (outFileName);
for(i=0;i<=maxNode; i++){
    outfile << i << "," << r2[i] << endl;
}
outfile.close();
}
