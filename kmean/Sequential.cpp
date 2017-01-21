#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include "structs.h"

#define PI 3.14159265

InitPoint* readFile(char* fileName, GlobalVar *data);
void calcPointsInTime(GlobalVar data, InitPoint* initPointsArr, Point* pointsArr, double currentT);
void randomClaster(K_Mean* kmeanArr, GlobalVar data);
void pointClusterDistance(K_Mean* kmeanArr, GlobalVar data);
void minDistance(double* arr, GlobalVar data, int* indexMinCluster);
void newClusters(K_Mean* kmeanArr, GlobalVar data, int* indexMinCluster);
void randomPoints(K_Mean* kmeanArr, GlobalVar* data);

//int main(int argc, char *argv[])
int main2()
{
	char* fileName = "dataFile.txt";
	InitPoint* initPointsArr = nullptr;		//Initial points from file
	GlobalVar sharedData;					//
	Result* resultsArr = nullptr;			//Array of kmean result for each time
	K_Mean kmeanArr;				//Array of kmean for each time
	int numIterations;						//number of times to perform kmean according to time
	int i;

	initPointsArr = readFile(fileName, &sharedData);

	//ResultArr:
	numIterations = sharedData.T / sharedData.delta_t;
	resultsArr = (Result*)malloc(numIterations * sizeof(Result));
	for (i = 0; i < numIterations; i++){
		resultsArr[i].t = i*sharedData.delta_t;
	}

	//Initalize
	kmeanArr.pointsArr = (Point*)malloc(sharedData.sizePointsArr *sizeof(Point));
	calcPointsInTime(sharedData, initPointsArr, kmeanArr.pointsArr, resultsArr[0].t);
	//randomPoints(&kmeanArr, &sharedData);

	kmeanArr.clusterArr = (Point*)malloc(sharedData.sizeClusterArr * sizeof(Point));
	randomClaster(&kmeanArr, sharedData);

	
	/*
	printf("points\n");
	for (i = 0; i < sharedData.sizePointsArr; i++){
		printf("%d, %f, %f\n", i, kmeanArr.pointsArr[i].x, kmeanArr.pointsArr[i].y);
	}
	*/
	
	for (int k = 0; k < sharedData.limit; k++)  
	{
		/*
		printf("\n\n ---------- iteration no. %d ----------------\n", k);
		printf("\nclusters\n");
		for (i = 0; i < sharedData.sizeClusterArr; i++){
			printf("%d, %f, %f\n", i, kmeanArr.clusterArr[i].x, kmeanArr.clusterArr[i].y);
		}
		printf("\n");
		*/

		kmeanArr.distancePointerCluster = (double*)malloc(sizeof(double)*(sharedData.sizePointsArr)*(sharedData.sizeClusterArr));
		pointClusterDistance(&kmeanArr, sharedData);
		int* indexMinCluster = (int*)malloc(sizeof(int)*sharedData.sizePointsArr);
		minDistance(kmeanArr.distancePointerCluster, sharedData, indexMinCluster);
		
		/*
		printf("\n\ndistance\n");
		for (i = 0; i < sharedData.sizePointsArr; i++){
			printf("%d ", i);
			for (int j = 0; j < sharedData.sizeClusterArr; j++){
				printf("--%f--", kmeanArr.distancePointerCluster[i*sharedData.sizeClusterArr + j]);
			}
			printf(" %d\n", indexMinCluster[i]);
		}
		*/
		newClusters(&kmeanArr, sharedData, indexMinCluster);

		free(indexMinCluster);
		free(kmeanArr.distancePointerCluster);
	}

	//*-*save result*-*
	


	//free
	free(kmeanArr.pointsArr);
	free(kmeanArr.clusterArr);

	free(resultsArr);
	printf("The end");
	fflush(stdout);
	getchar();
	return 0;
}

void randomPoints(K_Mean* kmeanArr, GlobalVar* data)
{
	int i;
	srand(time_t(NULL));
	for (i = 0; i < data->sizePointsArr; i++){
		kmeanArr->pointsArr[i].x = rand() % data->sizePointsArr;
		kmeanArr->pointsArr[i].y = rand() % data->sizePointsArr;
	}

}

/**
* step 0: Initialize - points array
* Calculate points according to formula:
* xi =  ai + Ri * cos(2*pi*t/T)
* yi =  bi + Ri * sin(2*pi*t/T)
*/
void calcPointsInTime(GlobalVar data, InitPoint* initPointsArr, Point* pointsArr, double currentT)
{
	for (int i = 0; i < data.sizePointsArr; i++){
		pointsArr[i].x = initPointsArr[i].a + initPointsArr[i].r * cos(2.0 * PI * currentT / data.T);
		pointsArr[i].y = initPointsArr[i].b + initPointsArr[i].r * cos(2.0 * PI * currentT / data.T);
	}
}

/**
* step 0: Initialize - cluster array
* Choose random points to be initial clusters
*
*/
void randomClaster(K_Mean* kmeanArr, GlobalVar data)
{
	int i, randomNum;
	srand(time_t(NULL));
	for (i = 0; i < data.sizeClusterArr; i++){
		randomNum = rand() % data.sizePointsArr;
		kmeanArr->clusterArr[i].x = kmeanArr->pointsArr[randomNum].x;
		kmeanArr->clusterArr[i].y = kmeanArr->pointsArr[randomNum].y;
	}
}

/**
* step 1: 
* calculate distance between points and clusters
*
*/
void pointClusterDistance(K_Mean* kmeanArr, GlobalVar data)
{
	int i, j;
	int width = data.sizeClusterArr;
	for (i = 0; i < data.sizePointsArr; i++)
	{
		double pX, pY, cX, cY;
		pX = kmeanArr->pointsArr[i].x;
		pY = kmeanArr->pointsArr[i].y;
		for (j = 0; j < data.sizeClusterArr; j++)
		{
			cX = kmeanArr->clusterArr[j].x;
			cY = kmeanArr->clusterArr[j].y;
			kmeanArr->distancePointerCluster[i*width + j] = sqrt(pow(cX - pX, 2) + pow(cY - pY, 2));
		}
	}
}

/**
* step 2:
* Find the closest cluster for each point
* input: arr = matrix of ditance between point and each cluster
*		 indexMinCluster = array of the cluster with minimum distance to point
*/
void minDistance(double* arr, GlobalVar data, int* indexMinCluster)
{
	int i, j;
	int width = data.sizeClusterArr;
	for (i = 0; i < data.sizePointsArr; i++)
	{
		indexMinCluster[i] = 0;
		double min = arr[i*width];
		for (j = 1; j < data.sizeClusterArr; j++)
		{
			if (min > arr[i*width + j]){
				indexMinCluster[i] = j;
				min = arr[i*width + j];
			}
		}
	}
}

/**
* step 3:
* Calculate mean of groups of points --> new clusters
*
*/
void newClusters(K_Mean* kmeanArr, GlobalVar data, int* indexPointCluster)
{
	int i;
	int* sumPointsCluster = (int*)calloc(sizeof(int), data.sizeClusterArr);

	for (i = 0; i < data.sizeClusterArr; i++)
	{
		kmeanArr->clusterArr[i].x = 0;
		kmeanArr->clusterArr[i].y = 0;
	}

	//for (i = 0; i < data.sizeClusterArr; i++)
	//	printf("%d:  %f, %f\n", i, kmeanArr->clusterArr[i].x, kmeanArr->clusterArr[i].y);

	for (i = 0; i < data.sizePointsArr; i++)
	{
		int index = indexPointCluster[i];
		sumPointsCluster[index]++;
		kmeanArr->clusterArr[index].x += kmeanArr->pointsArr[i].x;
		kmeanArr->clusterArr[index].y += kmeanArr->pointsArr[i].y;
	}

	for (i = 0; i < data.sizeClusterArr; i++)
	{
		int sum = sumPointsCluster[i];
		if (sum){
			kmeanArr->clusterArr[i].x /= sum;
			kmeanArr->clusterArr[i].y /= sum;
		}
	}
	free(sumPointsCluster);


	//for (i = 0; i < data.sizeClusterArr; i++)
	//	printf("%d:  %f, %f\n", i, kmeanArr->clusterArr[i].x, kmeanArr->clusterArr[i].y);
}


InitPoint* readFile(char* fileName, GlobalVar *data)
{
	InitPoint* pointsArr;
	int i=0;	
	FILE* f;
	f = fopen(fileName, "r");

	if (!f)
	{
		printf("Failed to open file\n");
		return NULL;
	}
	fscanf(f, "%d %d %lf %lf %d", &data->sizePointsArr, &data->sizeClusterArr, &data->delta_t, &data->T, &data->limit);
	pointsArr = (InitPoint*)malloc((data->sizePointsArr)*sizeof(InitPoint));

	while (!feof(f))
	{
		fscanf(f, "%d %lf %lf %lf", &pointsArr[i].pid, &pointsArr[i].a, &pointsArr[i].b, &pointsArr[i].r);
		i++;
	}

	fclose(f);
	return pointsArr;
}