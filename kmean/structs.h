#pragma once

typedef struct InitPoint {
	double a;
	double b;
	double r;
} InitPoint;

typedef struct GlobalVar {
	double delta_t;
	double T;
	int sizePointsArr = 0;
	int sizeClusterArr = 0;
	int limit;
	int chunk;

} GlobalVar;

typedef struct Point {
	double x;
	double y;
} Point;

typedef struct K_Mean {
	Point* pointsArr = nullptr;
	Point* clusterArr = nullptr;
} K_Mean;

typedef struct Result {
	double t;
	double d = NULL;
	Point* clusterArr = nullptr;
} Result;

