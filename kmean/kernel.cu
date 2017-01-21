
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include "cuda.h"


#define MAX_NUM_THREADS 1024

__device__ int *resultIndexCluster;
__device__ Point *clusters, *points;

/*
* Finds the index of the closest cluster to point.
* Return an array of cluster indexes
*/
__global__ void minDistanceCluster(Point *clusters, Point *points, int* resultIndexCluster, GlobalVar data)
{
	double pX, pY, cX, cY;
	double currentD, minD = DBL_MAX;
	int pIndex;
	pIndex = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
	
	if (pIndex < data.sizePointsArr)
	{
		pX = points[pIndex].x;
		pY = points[pIndex].y;
		for (int i = 0; i < data.sizeClusterArr; i++)
		{
			cX = clusters[i].x;
			cY = clusters[i].y;
			currentD = (cX - pX) * (cX - pX) + (cY - pY) * (cY - pY);
			if (currentD < minD){
				minD = currentD;
				resultIndexCluster[pIndex] = i;
			}
		}
	}
}

/*
* memcopy clusters and call kernel
*/
void closestClusterToPoint(int* indexMinCluster, Point* clusterArr, GlobalVar data)
{
	int i;
	int numBlock = (int)ceil((double)data.sizePointsArr / MAX_NUM_THREADS);

	cudaError_t cudaStatus;
	dim3 dimGrid(numBlock);
	dim3 dimBlock(MAX_NUM_THREADS);

	//Copy cluster array to device:
	cudaStatus = cudaMemcpy(clusters, clusterArr, data.sizeClusterArr * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMemcpy (clusters) failed!\n");
		fflush(stdout);
		return;
	}

	minDistanceCluster <<<dimGrid, dimBlock >>>(clusters, points, resultIndexCluster, data);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "distanceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		fflush(stdout);
		return;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		fflush(stdout);
		return;
	}

	//Read result from device:
	cudaMemcpy(indexMinCluster, resultIndexCluster, data.sizePointsArr*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMemcpy failed!\n");
		fflush(stdout);
		return;
	}
}

/*
* Malloc and memcopy only once per iteration
*/
void prepForCuda(GlobalVar data)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.				///------------- should be according to device id?? --------------
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		fflush(stdout);
		return;
	}

	//allocate memory in device for cluster array:
	cudaStatus = cudaMalloc(&clusters, data.sizeClusterArr *sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMalloc (clusters) failed!\n");
		fflush(stdout);
		return;
	}

	//allocate memory in device for points array:
	cudaStatus = cudaMalloc(&points, data.sizePointsArr *sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMalloc (points) failed!\n");
		fflush(stdout);
		return;
	}

	//allocate memory in device for result:
	cudaStatus = cudaMalloc(&resultIndexCluster, data.sizePointsArr *sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMalloc failed!\n");
		fflush(stdout);
		return;
	}
}

//free memory from device:
void freeAllocationCuda()
{
	cudaFree(resultIndexCluster);
	cudaFree(clusters);
	cudaFree(points);
}

void copyPoints(Point* pointsArr, GlobalVar data)
{
	cudaError_t cudaStatus;

	//Copy points array to device:
	cudaStatus = cudaMemcpy(points, pointsArr, data.sizePointsArr * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaMemcpy (points) failed!\n");
		fflush(stdout);
		return;
	}
}