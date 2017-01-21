
#include "structs.h"

void closestClusterToPoint(int* indexMinCluster, Point* clusterArr, GlobalVar data);
void prepForCuda(GlobalVar data);
void freeAllocationCuda();
void copyPoints(Point* pointsArr, GlobalVar data);