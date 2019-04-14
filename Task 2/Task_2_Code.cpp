
#include "stdafx.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "mpi.h"
#include <vector>
#include <algorithm>

int malloc2dint(int ***array, int n, int m) {

	/* allocate the n*m contiguous items */
	int *p = (int *)malloc(n*m*sizeof(int));
	if (!p) return -1;

	/* allocate the row pointers into the memory */
	(*array) = (int **)malloc(n*sizeof(int*));
	if (!(*array)) {
		free(p);
		return -1;
	}

	/* set up the pointers into the contiguous memory */
	for (int i = 0; i<n; i++)
		(*array)[i] = &(p[i*m]);

	return 0;
}

bool mySortFunction(const std::vector<int>& inner1, const std::vector<int>& inner2) {
	return inner1[0] < inner2[0];
}


int main(int argc, char* argv[]){

	/*** Specify input filename (must be in same directory) ***/
	const char* fileName = "100000_key-value_pairs.csv";

	/*** Specify output filename ***/
	const char* outFileName = "Output_Task2.csv";

	int  my_rank;    /* rank of process */
	int  size;       /* number of processes */
	int tag = 0;     /* tag for messages */
	int maxKey;
	int minKey;

	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	//============================================================================================//
	// READING FILE & LOADING DATA                                                                //
	//============================================================================================//
	if (my_rank == 0) printf("Reading file...\n");
	
	/*** Count number of lines in file ***/
	std::string line;
	std::ifstream infile(fileName);
	int number_of_lines = 0;
	while (getline(infile, line)){
		++number_of_lines;
	}
	//std::cout << "Number of lines in file: " << number_of_lines << std::endl << std::endl;
	
	int **global=NULL;
	int n = number_of_lines;
	int m = 2;
	if (my_rank == 0){
		/*** Store keys and values from file: data=[key][value] ***/

		malloc2dint(&global, n, m);
		std::string line1;
		std::string line2;
		std::ifstream myfile(fileName);

		int i = 0;
		int k = 0;
		// Grab entire line //
		while (getline(myfile, line1)){
			std::istringstream iss(line1);

			// Split line at custom delimiter //
			i = 0;
			while (getline(iss, line2, ',')){
				// Do not store data labels //
				if (line2 == "key" || line2 == "value"){
					k--;
					break;
				}
				else{
					if (i == 0){
						global[k][0] = atoi(line2.c_str());
					}
					else{
						global[k][1] = atoi(line2.c_str());
					}
					i++;
				}
			}
			k++;
		}

		// Calculate range of keys //
		maxKey = global[0][0];
		minKey = global[0][0];
		for (int i = 0; i < number_of_lines - 1; i++){
			if (global[i][0] > maxKey) maxKey = global[i][0];
			if (global[i][0] < minKey) minKey = global[i][0];
		}
		//std::cout << "Max Key:" << maxKey << std::endl;
		//std::cout << "Min Key:" << minKey << std::endl;
		
	}

	// Broadcast min and max key values from root to other processes //
	MPI_Bcast(&maxKey, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&minKey, 1, MPI_INT, 0, MPI_COMM_WORLD);


	//if (my_rank == 0){
	//	printf("Global array is:\n");
	//	for (int i = 0; i<number_of_lines-1; i++) {
	//		for (int j = 0; j<2; j++)
	//			std::cout<<(global[i][j]) << "\t";

	//		printf("\n");
	//	}
	//}

	//============================================================================================//
	// SCATTER DATA TO ALL PROCESSES                                                              //
	//============================================================================================//
	if (my_rank == 0) printf("Scattering data to nodes...\n");

	/*** Create a datatype to describe the subarrays of the global array ***/
	const int numProc = size;
	const int rows = number_of_lines - 1;
	int sizes[2] = {rows, 2};                        /* global size */
	int rem = rows % numProc;				         /* remainder work after equal distribution */
	int subsizes[2] = { rows / numProc, 2 };         /* subarray size  */
	
	int starts[2] = { 0, 0 };                  /* start point */
	MPI_Datatype type, subarrtype;
	MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &type);
	MPI_Type_create_resized(type, 0, 2*subsizes[0]*sizeof(int), &subarrtype);
	MPI_Type_commit(&subarrtype);

	/*** Allocate memory for the local array ***/
	int **local;
	malloc2dint(&local, (numProc-1)*subsizes[0], 2);
	
	int *globalptr = NULL;
	if (my_rank == 0) globalptr = &(global[0][0]);

	/*** Calculate number of blocks to send to each processor and displacments ***/
	int* sendcounts = 0;
	int* displs = 0;
	sendcounts = new int[numProc];
	displs = new int[numProc];
	if (my_rank == 0) {
		for (int i = 0; i < numProc; i++) sendcounts[i] = 1;	// Distribute one block per process //
		if (rem != 0) sendcounts[numProc - 1] = (numProc-1);	            // Add extra blocks to last processor if necessary //
		for (int i = 0; i < numProc; i++) {
			displs[i] = i;
		}
	}

	/*** Send scatter communication ***/
	MPI_Scatterv(globalptr, sendcounts, displs, subarrtype, &(local[0][0]),
		(numProc-1)*(2*subsizes[0]), MPI_INT,
		0, MPI_COMM_WORLD);

	/*** Print local data ***/
	for (int p = 0; p<size; p++) {
		if (my_rank == p) {
			int localRows = subsizes[0];
			if (p == numProc - 1) localRows = subsizes[0] + rem;
			//printf("Local process on rank %d is:\n", my_rank);
			//for (int i = 0; i<localRows; i++) {
			//	putchar('|');
			//	for (int j = 0; j<2; j++) {
			//		std::cout<<local[i][j] << "\t";
			//	}
			//	printf("|\n");
			//}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	//============================================================================================//
	// LOCAL REDUCE 1                                                                             //
	//============================================================================================//
	if (my_rank == 0) printf("Local reduce 1...\n");

	// Allocate memory for reduced local array //
	int **redLocal;
	malloc2dint(&redLocal, (numProc - 1)*subsizes[0], 2);

	int s;
	for (int p = 0; p<size; p++) {
		if (my_rank == p) {
			int localRows = subsizes[0];
			if (p == numProc - 1) localRows = subsizes[0] + rem;

			// Convert array to vector //
			std::vector<std::vector<int> > vect;
			for (int i = 0; i < localRows; ++i) {
				std::vector<int> inner(local[i], local[i] + 2);
				vect.push_back(inner);
			}

			// Sort vector //
			sort(vect.begin(), vect.end(), mySortFunction);

			// Covert vector back to array //
			for (int i = 0; i < localRows; ++i) {
				local[i][0] = vect[i][0];
				local[i][1] = vect[i][1];
			}

			//// Print sorted array //
			//printf("SORTED: Local process on rank %d is:\n", my_rank);
			//for (int i = 0; i<localRows; i++) {
			//	putchar('|');
			//	for (int j = 0; j<2; j++) {
			//		std::cout << local[i][j] << "\t";
			//	}
			//	printf("|\n");
			//}

			// Perform local reduce on sorted array //
			int k;
			s = 0;            /* index for reduced local array */
			for (int i = 0; i < localRows; i++){
				
				// Find values i through k that share same key // 
				k = i;
				while (k + 1 < rows) {
					if (local[k][0] == local[k + 1][0]){
						k++;
					}
					else break;
				}
				//k = i;
				//while (local[k][0] == local[k + 1][0]){
				//	k++;
				//}
				
				// Sum values i through k //
				redLocal[s][0] = local[i][0];
				redLocal[s][1] = local[i][1];
				if (k != i) {
					for (int j = i + 1; j <= k; j++) redLocal[s][1] += local[k][1];
				}

				i = k; /* skip to next unique key */
				s++;  /* increment reduced local arrray index */
			}

			//// Print reduced array //
			//printf("REDUCED: Local process on rank %d is:\n", my_rank);
			//for (int i = 0; i<s; i++) {
			//	putchar('|');
			//	for (int j = 0; j<2; j++) {
			//		std::cout << redLocal[i][j] << "\t";
			//	}
			//	printf("|\n");
			//}

		}
	}

	//============================================================================================//
	// ALL TO ALL PERSONALIZED                                                                    //
	//============================================================================================//
	if (my_rank == 0) printf("All to all personalized...\n");

	int *sdisp, *scounts, *rdisp, *rcounts;
	int i;

	// Allocate memory for counts and displacements //
	scounts = (int*)malloc(sizeof(int)*numProc);		/* send counts for each process */
	rcounts = (int*)malloc(sizeof(int)*numProc);		/* receive counts for each process */
	sdisp = (int*)malloc(sizeof(int)*numProc);			/* send displacements */
	rdisp = (int*)malloc(sizeof(int)*numProc);			/* receive displacements */

	// Calculate the keys to send to each process //
	int keyInt = (maxKey - minKey) / numProc;
	int keyRem = (maxKey - minKey) % numProc;
	int *keySet;
	keySet = new int[numProc];
	keySet[0] = minKey;
	for (int i = 1; i<size; i++) {
		keySet[i] = keySet[i - 1] + keyInt;				/* equal distribution of keys */
	}
	keySet[size] = maxKey+1;								/* last process takes all remaining keys */
	//printf("myid= %d keySet=%d %d %d %d %d\n", my_rank, keySet[0], keySet[1], keySet[2], keySet[3], keySet[4]);
	//printf("myid= %d maxKey=%d\n", my_rank, maxKey);

	// Find out how much data to send //
	for (int p = 0; p < size; p++) {
		if (my_rank == p) {

			int ind = 0;
			for (int i = 0; i < size; i++){
				scounts[i] = 0;
				while (redLocal[ind][0] < keySet[i + 1]){
					if (ind < s) {
						scounts[i]+=2;
						ind++;
					}
					else break;
				}
			}
			//printf("myid= %d scounts=%d %d %d %d\n", my_rank, scounts[0], scounts[1], scounts[2], scounts[3]);
		}
	}

	// Tell the other processors how much data is coming (MPI_AlltoAll) //
	MPI_Alltoall(scounts, 1, MPI_INT,rcounts, 1, MPI_INT,MPI_COMM_WORLD);
	//printf("myid= %d rcounts=%d %d %d %d\n", my_rank, rcounts[0], rcounts[1], rcounts[2], rcounts[3]);

	// Calculate displacements and the size of the arrays //
	sdisp[0] = 0;
	for (i = 1; i<numProc; i++){
		sdisp[i] = scounts[i - 1] + sdisp[i - 1];
	}
	//printf("myid= %d sdisp=%d %d %d %d\n", my_rank, sdisp[0], sdisp[1], sdisp[2], sdisp[3]);
	rdisp[0] = 0;
	for (i = 1; i<numProc; i++){
		rdisp[i] = rcounts[i - 1] + rdisp[i - 1];
	}
	//printf("myid= %d rdisp=%d %d %d %d\n", my_rank, rdisp[0], rdisp[1], rdisp[2], rdisp[3]);

	// Allocate memory for local array //
	int rows2 = 0;
	for (int i = 0; i < numProc; i++) rows2 += rcounts[i]/2;
	int **local2;
	malloc2dint(&local2, rows2, 2);

	// Send/rec different amounts of data to/from each processor //
	int *myptr = NULL;
	myptr = &(redLocal[0][0]);
	int *myptr2 = NULL;
	myptr2 = &(local2[0][0]);
	MPI_Alltoallv(myptr, scounts, sdisp, MPI_INT, myptr2, rcounts, rdisp, MPI_INT,
		MPI_COMM_WORLD);

	//// Print array //
	//printf("LOCAL 2: Local process on rank %d is:\n", my_rank);
	//for (int i = 0; i<rows2; i++) {
	//	putchar('|');
	//	for (int j = 0; j<2; j++) {
	//		std::cout << local2[i][j] << "\t";
	//	}
	//	printf("|\n");
	//}

	//============================================================================================//
	// LOCAL REDUCE 2                                                                             //
	//============================================================================================//
	if (my_rank == 0) printf("Local reduce 2...\n");

	// Allocate memory for reduced local array //
	int **redLocal2;
	malloc2dint(&redLocal2, rows2, 2);

	int s2;
	for (int p = 0; p<size; p++) {
		if (my_rank == p) {

			// Convert array to vector //
			std::vector<std::vector<int> > vect2;
			for (int i = 0; i < rows2; ++i) {
				std::vector<int> inner(local2[i], local2[i] + 2);
				vect2.push_back(inner);
			}

			// Sort vector //
			sort(vect2.begin(), vect2.end(), mySortFunction);

			// Covert vector back to array //
			for (int i = 0; i < rows2; ++i) {
				local2[i][0] = vect2[i][0];
				local2[i][1] = vect2[i][1];
			}

			//// Print sorted array //
			//printf("SORTED 2: Local process on rank %d is:\n", my_rank);
			//for (int i = 0; i<rows2; i++) {
			//	putchar('|');
			//	for (int j = 0; j<2; j++) {
			//		std::cout << local2[i][j] << "\t";
			//	}
			//	printf("|\n");
			//}

			// Perform local reduce on sorted array //
			s2 = 0;            /* index for reduced local array */
			int k;
			for (int i = 0; i < rows2; i++){

				// Find values i through k that share same key // 
				k = i;
				while (k + 1 < rows2) {
					if (local2[k][0] == local2[k + 1][0]){
						k++;
					}
					else break;
				}

				// Sum values i through k //
				redLocal2[s2][0] = local2[i][0];
				redLocal2[s2][1] = local2[i][1];
				if (k != i) {
					for (int j = i + 1; j <= k; j++) redLocal2[s2][1] += local2[k][1];
				}

				i = k; /* skip to next unique key */
				s2++;  /* increment reduced local arrray index */
			}

			//// Print reduced array //
			//printf("REDUCED 2: Local process on rank %d is:\n", my_rank);
			//for (int i = 0; i<s2; i++) {
			//	putchar('|');
			//	for (int j = 0; j<2; j++) {
			//		std::cout << redLocal2[i][j] << "\t";
			//	}
			//	printf("|\n");
			//}

		}
	}

	//============================================================================================//
	// GATHER                                                                                     //
	//============================================================================================//
	if (my_rank == 0) printf("Gather data to master node...\n");

	// Allocate memory //
	int *counts2, *displacements;
	if (my_rank == 0){
		counts2 = (int*)malloc(numProc*sizeof(int));
		displacements = (int*)malloc(numProc*sizeof(int));
	}

	// Calculate counts for each local array //
	int subCounts = 0;
	for (int p = 0; p < size; p++) {
		if (my_rank == p) {
			subCounts = 2 * s2;
			//printf("Rank %d, counts: %d\n", my_rank,subCounts);
		}
	}

	// Gather all counts to root //
	MPI_Gather(&subCounts, 1, MPI_INT, counts2, 1, MPI_INT,
		0, MPI_COMM_WORLD);
	//if (my_rank == 0) printf("myid= %d gcounts=%d %d %d %d\n", my_rank, counts2[0], counts2[1], counts2[2], counts2[3]);

	// Calculate displacements //
	if (my_rank == 0){
		displacements[0] = 0;
		for (i = 1; i<numProc; i++){
			displacements[i] = counts2[i - 1] + displacements[i - 1];
		}
	}
	//if (my_rank == 0) printf("myid= %d disp=%d %d %d %d\n", my_rank, displacements[0], displacements[1], displacements[2], displacements[3]);

	// Allocate memory for global array //
	int **redGlobal = NULL;
	int globRows = 0;
	if (my_rank == 0){
		for (int p = 0; p < size; p++) globRows += counts2[p] / 2;
		malloc2dint(&redGlobal, globRows, 2);
	}

	// Gather all local arrays to root //
	int *globalptr2 = NULL;
	int *localptr = NULL;
	localptr = &(redLocal2[0][0]);
	if (my_rank == 0) globalptr2 = &(redGlobal[0][0]);
	MPI_Gatherv(localptr, subCounts, MPI_INT, globalptr2, counts2,
		displacements, MPI_INT, 0, MPI_COMM_WORLD);

	//// Print global array //
	//if (my_rank == 0){
	//	printf("Global array is:\n");
	//	for (int i = 0; i<globRows; i++) {
	//		for (int j = 0; j<2; j++)
	//			std::cout<<(redGlobal[i][j]) << "\t";

	//		printf("\n");
	//	}
	//}


	//============================================================================================//
	// WRITE TO FILE                                                                              //
	//============================================================================================//
	if (my_rank == 0) printf("Writing file...\n");

	/*** Write results to file ***/
	if (my_rank == 0){
		std::ofstream outfile;
		outfile.open(outFileName);
		outfile << "key" << "," << "value" << "\n";
		for (int i = 0; i<globRows; i++) {
			for (int j = 0; j<2; j++)
				outfile<< (redGlobal[i][j]) << ",";

			outfile << "\n";
		}
		outfile.close();
	}

	if (my_rank == 0) printf("Complete!!\n");
	MPI_Finalize();
	return 0;
}


