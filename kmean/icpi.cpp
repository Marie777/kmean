
/*
*  
*      
*/

#include<mpi.h>
#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<float.h>
#include "cuda.h"
#include "structs.h"

#define PI 3.14159265

void master(int numprocs, MPI_Datatype* GlobalVarMPIType, MPI_Datatype* InitPointMPIType, MPI_Datatype* PointMPIType);
void slave(MPI_Datatype* GlobalVarMPIType, MPI_Datatype* InitPointMPIType, MPI_Datatype* PointMPIType);
double* createIterationTimeArr(GlobalVar sharedData, int* numIterations);
void sendTaskSlave(double* iterationTArr, int slaveId, int* iterationCount, int numIterations, int* taskCount, int chunk);
InitPoint* readFile(char* fileName, GlobalVar *data);
void writeResultFile(double* td, Point* clusters, int numOfClusters);
Point* calcPointsInTime(GlobalVar data, InitPoint* initPointsArr, double currentT);
Point* randomClaster(Point* pointsArr, GlobalVar data);
Point* newClusters(Point* pointsArr, Point* clusterArr, GlobalVar data, int* indexPointCluster, int* keepGoing);
Result kmeanOverTime(InitPoint* initPointsArr, GlobalVar sharedData, double* iterationTArr, int chunk);
double min_d_clusters(Point* arr, int size);
int sizeChunk(int numIterations);
void printPoints(Point* arr, int size);
void globalVarCreateType(MPI_Datatype* GlobalVarMPIType);
void initPointCreateType(MPI_Datatype* InitPointMPIType);
void pointCreateType(MPI_Datatype* PointMPIType);


int main(int argc, char *argv[])
{
	int myId, numprocs;
	double startTime, endTime;
	MPI_Datatype GlobalVarMPIType;
	MPI_Datatype InitPointMPIType;
	MPI_Datatype PointMPIType;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	globalVarCreateType(&GlobalVarMPIType);
	initPointCreateType(&InitPointMPIType);
	pointCreateType(&PointMPIType);
	
	if (numprocs < 2 ){
		printf("numprocs should be more than 1");
		return 0;
	}

	if (myId == 0){
		startTime = MPI_Wtime();
		master(numprocs, &GlobalVarMPIType, &InitPointMPIType, &PointMPIType);
		endTime = MPI_Wtime();
		printf("Total time: %f\n", endTime - startTime);
	}else{
		slave(&GlobalVarMPIType, &InitPointMPIType, &PointMPIType);
	}

	MPI_Finalize();

	//printf("Id %d Finished\n",myId);
	fflush(stdout);
	return 0;
}


/** 
* MPI: Master sends initial data and distributes jobs to slaves (load balancing)
*      The master receives results from slaves and saves the min result to file.
*/
void master(int numprocs, MPI_Datatype* GlobalVarMPIType, MPI_Datatype* InitPointMPIType, MPI_Datatype* PointMPIType)
{
	MPI_Status status;
	InitPoint* initPointsArr;		//Initial points from file
	GlobalVar sharedData;					//constant varibales from file
	Point* currentCluster = nullptr;
	Point* finalCluster = nullptr;
	char* fileName = "pointFile.txt";
	int i, numIterations, iterationCount = 0, taskCount = 0;
	double* iterationTArr, td[2], finaltd[2];

	initPointsArr = readFile(fileName, &sharedData);
	iterationTArr = createIterationTimeArr(sharedData, &numIterations);
	sharedData.chunk = sizeChunk(numIterations);
	currentCluster = (Point*)malloc(sizeof(Point)*sharedData.sizeClusterArr);
	if (!currentCluster){
		printf("Failed to allocate memory for currentCluster!");
		fflush(stdout); exit(1);
	}

	for (i = 1; i < numprocs; i++){
		MPI_Send(&sharedData, 1, *GlobalVarMPIType, i, 0, MPI_COMM_WORLD);
		MPI_Send(initPointsArr, sharedData.sizePointsArr, *InitPointMPIType, i, 0, MPI_COMM_WORLD);
		sendTaskSlave(iterationTArr, i, &iterationCount, numIterations, &taskCount, sharedData.chunk);
	}
	//Load balancing:
	finaltd[1] = DBL_MAX;
	do{
		MPI_Recv(&td, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(currentCluster, sharedData.sizeClusterArr, *PointMPIType, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		taskCount--;
		if (td[1] < finaltd[1]){
			finalCluster = currentCluster;
			finaltd[1] = td[1];
			finaltd[0] = td[0];
		}
		sendTaskSlave(iterationTArr, status.MPI_SOURCE, &iterationCount, numIterations, &taskCount, sharedData.chunk);
	} while (taskCount > 0);

	//sent stop recv to slaves
	for (i = 1; i < numprocs; i++){
		MPI_Send(&i, 0, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
	}
	free(iterationTArr);
	free(initPointsArr);
	writeResultFile(finaltd, finalCluster, sharedData.sizeClusterArr);
	free(currentCluster);
}

/*
* Create array of iterations according to delta t and T
*/
double* createIterationTimeArr(GlobalVar sharedData, int* numIterations)
{
	double* iterationTArr;
	*numIterations = sharedData.T / sharedData.delta_t;
	iterationTArr = (double*)malloc(*numIterations * sizeof(double));
	if (!iterationTArr) {
		printf("Failed to allocate memory for iterationTArr!");
		fflush(stdout); exit(1);
	}
	for (int i = 0; i < *numIterations; i++)
		iterationTArr[i] = i*sharedData.delta_t;

	return iterationTArr;
}

/*
* Send iterations to slaves according to chunk
*/
void sendTaskSlave(double* iterationTArr, int slaveId, int* iterationCount, int numIterations, int* taskCount, int chunk)
{
	if (*iterationCount < numIterations){
		MPI_Send(&iterationTArr[*iterationCount], chunk, MPI_DOUBLE, slaveId, 0, MPI_COMM_WORLD);
		(*iterationCount) += chunk;
		(*taskCount)++;
	}
}


/**
* MPI: Slave receives data from master and calc kmean over time.
*	   The slave sends back min result.
*/
void slave(MPI_Datatype* GlobalVarMPIType, MPI_Datatype* InitPointMPIType, MPI_Datatype* PointMPIType)
{
	MPI_Status status;
	InitPoint* initPointsArrS;
	GlobalVar sharedDataS;
	Result resultS;
	double* iterationTArrS;
	double td[2];

	MPI_Recv(&sharedDataS, 1, *GlobalVarMPIType, 0, 0, MPI_COMM_WORLD, &status);
	initPointsArrS = (InitPoint*)malloc(sizeof(InitPoint)* sharedDataS.sizePointsArr);
	if (!initPointsArrS) {
		printf("Failed to allocate memory for initPointsArrS!\n");
		fflush(stdout); exit(1);
	}
	MPI_Recv(initPointsArrS, sharedDataS.sizePointsArr, *InitPointMPIType, 0, 0, MPI_COMM_WORLD, &status);
	iterationTArrS = (double*)malloc(sizeof(double)*sharedDataS.chunk);
	if (!iterationTArrS) {
		printf("Failed to allocate memory for iterationTArrS!\n");
		fflush(stdout); exit(1);
	}

	while (1){
		MPI_Recv(iterationTArrS, sharedDataS.chunk, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG != 0)
			break;
		resultS = kmeanOverTime(initPointsArrS, sharedDataS, iterationTArrS, sharedDataS.chunk);
		td[0] = resultS.t;
		td[1] = resultS.d;
		MPI_Send(td, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Send(resultS.clusterArr, sharedDataS.sizeClusterArr, *PointMPIType, 0, 0, MPI_COMM_WORLD);
		free(resultS.clusterArr);
	}
	free(initPointsArrS);
	free(iterationTArrS);
}

/*
* Calculate kmean according to time
*/
Result kmeanOverTime(InitPoint* initPointsArr, GlobalVar sharedData, double* iterationTArr, int chunk)
{
	Result result;
	result.d = DBL_MAX;
	result.clusterArr = nullptr;
	int keepGoing = 1;

	prepForCuda(sharedData);
	for (int it = 0; it < chunk; it++){
		Point* pointsArr = calcPointsInTime(sharedData, initPointsArr, iterationTArr[it]);
		Point* clusterArr = randomClaster(pointsArr, sharedData);

		//Malloc for index of closest cluster to point
		int* indexMinCluster = (int*)malloc(sizeof(int)*sharedData.sizePointsArr);
		if (!indexMinCluster) {
			printf("Failed to allocate memory for indexMinCluster!");
			fflush(stdout); exit(1);
		}

		copyPoints(pointsArr, sharedData);
		for (int k = 0; k < sharedData.limit && keepGoing; k++)
		{	
			closestClusterToPoint(indexMinCluster, clusterArr, sharedData);
			clusterArr = newClusters(pointsArr, clusterArr, sharedData, indexMinCluster, &keepGoing); 
		}

		//Keep only the min result --> compare the last point in time result to the current result
		double current_d = min_d_clusters(clusterArr, sharedData.sizeClusterArr);

		if (result.d > current_d){
			Point* temp = result.clusterArr;
			result.clusterArr = clusterArr;
			if (!it){
				free(temp);
			}
			result.d = current_d;
			result.t = iterationTArr[it];
		}else{
			free(clusterArr);
		}

		free(indexMinCluster);
		free(pointsArr);
	}
	freeAllocationCuda();
	return result;
}


/**
* step 0: Initialize - points array
* Calculate points according to formula:
* xi =  ai + Ri * cos(2*pi*t/T)
* yi =  bi + Ri * sin(2*pi*t/T)
*/
Point* calcPointsInTime(GlobalVar data, InitPoint* initPointsArr, double currentT)
{
	Point* pointsArr = (Point*)malloc(data.sizePointsArr *sizeof(Point));
	if (!pointsArr) {
		printf("Failed to allocate memory for kmean.pointsArr!");
		fflush(stdout); exit(1);
	}
#pragma omp parallel for
	for (int i = 0; i < data.sizePointsArr; i++){
		pointsArr[i].x = initPointsArr[i].a + initPointsArr[i].r * cos(2.0 * PI * currentT / data.T);
		pointsArr[i].y = initPointsArr[i].b + initPointsArr[i].r * cos(2.0 * PI * currentT / data.T);
	}
	return pointsArr;
}

/**
* step 0: Initialize - cluster array
* Choose random points to be initial clusters
*
*/
Point* randomClaster(Point* pointsArr, GlobalVar data)
{
	int randomNum;

	Point* clusterArr = (Point*)malloc(data.sizeClusterArr * sizeof(Point));
	if (!clusterArr) {
		printf("Failed to allocate memory for kmean.clusterArr!");
		fflush(stdout); exit(1);
	}
	int* used = (int*)calloc(sizeof(int), data.sizePointsArr);
	if (!used) {
		printf("Failed to allocate memory for used!");
		fflush(stdout); exit(1);
	}
	srand(time_t(NULL));
	for (int i = 0; i < data.sizeClusterArr; i++){
		do{
			randomNum = rand() % data.sizePointsArr;
		} while (used[randomNum] != 0);
		clusterArr[i].x = pointsArr[randomNum].x;
		clusterArr[i].y = pointsArr[randomNum].y;
		used[randomNum]++;
	}
	free(used);
	return clusterArr;
}


/**
* step 3:
* Calculate mean of groups of points --> new clusters
*
*/
Point* newClusters(Point* pointsArr, Point* clusterArr, GlobalVar data, int* indexPointCluster, int* keepGoing)
{
	int i;
	
	int* sumPointsCluster = (int*)calloc(sizeof(int), data.sizeClusterArr);
	if (!sumPointsCluster) {
		printf("Failed to allocate memory for sumPointsCluster!");
		fflush(stdout);
		exit(1);
	}	
	Point* newCluster = (Point*)calloc(sizeof(Point), data.sizeClusterArr);
	if (!newCluster) {
		printf("Failed to allocate memory for newCluster!");
		fflush(stdout);
		exit(1);
	}

#pragma omp parallel for
	for (i = 0; i < data.sizePointsArr; i++)
	{
		int index = indexPointCluster[i];
		sumPointsCluster[index]++;
		newCluster[index].x += pointsArr[i].x;
		newCluster[index].y += pointsArr[i].y;
	}

#pragma omp parallel for
	for (i = 0; i < data.sizeClusterArr; i++)
	{
		int sum = sumPointsCluster[i];
		if (sum){
			newCluster[i].x /= sum;
			newCluster[i].y /= sum;
		}
	}
	free(sumPointsCluster);

	*keepGoing = 0;
	for (i = 0; i < data.sizeClusterArr; i++)
	{
		if (newCluster[i].x != clusterArr[i].x || newCluster[i].y != clusterArr[i].y)
		{
			*keepGoing = 1;
			break;
		}
	}
	free(clusterArr);
	return newCluster;
}

/*
* Calculates min distance between clusters
*/
double min_d_clusters(Point* arr, int size)
{
	double current_d, min_d = DBL_MAX, x1, x2, y1, y2;
		
	for (int i = 0; i < size; i++){
		x1 = arr[i].x;
		y1 = arr[i].y;
		for (int j = 0; j < size; j++){
			if (i != j){				//avoid calc d between cluster to itself
				x2 = arr[j].x;
				y2 = arr[j].y;
				current_d = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
				if (current_d < min_d){
					min_d = current_d;
				}
			}
		}
	}
	return min_d;
}

/*
* Read initial points from file 
*/
InitPoint* readFile(char* fileName, GlobalVar *data)
{
	InitPoint* pointsArr;
	int i = 0;
	FILE* f;
	f = fopen(fileName, "r");

	if (!f){
		printf("Failed to open file\n");
		return NULL;
	}
	fscanf(f, "%d %d %lf %lf %d", &data->sizePointsArr, &data->sizeClusterArr, &data->delta_t, &data->T, &data->limit);
	pointsArr = (InitPoint*)malloc((data->sizePointsArr)*sizeof(InitPoint));

	if (!pointsArr) {
		printf("Failed to allocate memory for pointsArr!");
		fflush(stdout); exit(1);
	}
	while (!feof(f)){
		int pid;
		fscanf(f, "%d %lf %lf %lf", &pid, &pointsArr[i].a, &pointsArr[i].b, &pointsArr[i].r);
		i++;
	}
	fclose(f);
	return pointsArr;
}

/*
* Write result to file
*/
void writeResultFile(double* td, Point* clusters, int numOfClusters)
{
	FILE* f;
	f = fopen("result.txt", "w+");
	if (!f){
		printf("Failed to open file\n");
		return;
	}
	fprintf(f,"d = %f\nt = %f \n centers of clusters:\n", td[1], td[0]);
	for (int i = 0; i < numOfClusters; i++){
		fprintf(f, "%f %f\n", clusters[i].x, clusters[i].y);
	}
	fclose(f);
}

/*
* choose how many iteration will be sent from master to slave each time.
* the function insures: number iterations % chunk == 0
*/
int sizeChunk(int numIterations)
{
	int i, chunk;
	for (i = 10; i > 0; i--){
		chunk = i * 10;
		if ((numIterations % (i * 10)) == 0 && chunk != numIterations && chunk < numIterations)
			return chunk;
	}
	for (i = 10; i > 2; i--){
		chunk = i;
		if ((numIterations % i) == 0 && chunk != numIterations && chunk < numIterations)
			return chunk;
	}
	chunk = numIterations; //the number of iterations is a prime number
	return chunk;
}
void printPoints(Point* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d: %f %f\n",i, arr[i].x, arr[i].y);
	}
}

void pointCreateType(MPI_Datatype* PointMPIType)
{
	Point point;
	MPI_Datatype typePoint[2] = { MPI_DOUBLE, MPI_DOUBLE };
	int blocklenPoint[2] = { 1, 1 };
	MPI_Aint dispPoint[2];

	dispPoint[0] = (char *)&point.x - (char *)&point;
	dispPoint[1] = (char *)&point.y - (char *)&point;
	MPI_Type_create_struct(2, blocklenPoint, dispPoint, typePoint, PointMPIType);
	MPI_Type_commit(PointMPIType);
}

void globalVarCreateType(MPI_Datatype* GlobalVarMPIType)
{
	GlobalVar sharedData;
	MPI_Datatype typeGlobalVar[6] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT, MPI_INT };
	int blocklenGlobalVar[6] = { 1, 1, 1, 1, 1, 1 };
	MPI_Aint dispGlobalVar[6];

	dispGlobalVar[0] = (char *)&sharedData.delta_t - (char *)&sharedData;
	dispGlobalVar[1] = (char *)&sharedData.T - (char *)&sharedData;
	dispGlobalVar[2] = (char *)&sharedData.sizePointsArr - (char *)&sharedData;
	dispGlobalVar[3] = (char *)&sharedData.sizeClusterArr - (char *)&sharedData;
	dispGlobalVar[4] = (char *)&sharedData.limit - (char *)&sharedData;
	dispGlobalVar[5] = (char *)&sharedData.chunk - (char *)&sharedData;
	MPI_Type_create_struct(6, blocklenGlobalVar, dispGlobalVar, typeGlobalVar, GlobalVarMPIType);
	MPI_Type_commit(GlobalVarMPIType);

}

void initPointCreateType(MPI_Datatype* InitPointMPIType)
{
	struct InitPoint* initPointsArr;
	MPI_Datatype typeInitPoint[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklenInitPoint[3] = { 1, 1, 1 };
	MPI_Aint dispInitPoint[3];

	dispInitPoint[0] = (char *)&initPointsArr->a - (char *)initPointsArr;
	dispInitPoint[1] = (char *)&initPointsArr->b - (char *)initPointsArr;
	dispInitPoint[2] = (char *)&initPointsArr->r - (char *)initPointsArr;
	MPI_Type_create_struct(3, blocklenInitPoint, dispInitPoint, typeInitPoint, InitPointMPIType);
	MPI_Type_commit(InitPointMPIType);

}
