
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <cuda.h>
#include <stdbool.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cfloat>
#include <fstream>
#include <sstream>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <time.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define P 2
#define K 233
#define N 35067
#define TOL 0.000005
#define SEED 1234ULL
#define TD 1024

#define CURAND_CALL(x) do { \
	if ((x)!=CURAND_STATUS_SUCCESS) { \
		printf("ERROR AT %s:%d\r\n", __FILE__, __LINE__);\
		return EXIT_FAILURE;\
	}\
} while(0)

using namespace std;

__global__ void initializeVectorToValue(float *vector, float value, int bound) {
	int tx = threadIdx.x;
	int bdx = blockDim.x;
	int bx = blockIdx.x;
	int index = bx * bdx + tx;
	if (index < bound) {
		vector[index] = value;
	}
}

__global__ void initializeAssignment(short *assignment, int n) {
	short index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < K) {
		assignment[index] = index;
	}
}
__global__ void initializeMean(float *mean, float* dataset, int n,
		int P) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bdx = blockDim.x;
	int bdy = blockDim.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int row = by * bdy + ty;
	int col = bx * bdx + tx;
	if (row < K && col < P) {
		mean[row * P + col] = dataset[row * P + col];
	}
}

__global__ void computeSum37(float* dataset, short* devAssignment,
		float* centroids, int* counter) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bdx = blockDim.x;
	int bdy = blockDim.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int row = by * bdy + ty;
	int col = bx * bdx + tx;
	if (row < N) {
		int clusterIndex = devAssignment[row];
		atomicAdd(&(centroids[clusterIndex * P + col]),
				dataset[row * P + col]);
		atomicAdd(&(counter[clusterIndex]), 1);
	}

}

__global__ void computeMean37(float* centroids, int* counter) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bdx = blockDim.x;
	int bdy = blockDim.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int row = by * bdy + ty;
	int col = bx * bdx + tx;

	if (row < K) {
		centroids[row * P + col] = centroids[row * P + col]
				/ (counter[row] / 2);
	}
}

__global__ void computeMean(float *centroids, float* dataset,
		short* devAssignment, int n, int P) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bdx = blockDim.x;
	int bdy = blockDim.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int row = by * bdy + ty;
	int col = bx * bdx + tx;
	float mean = 0.0;
	int counter = 0;
	if (row < K) {
		for (int i = 0; i < n; i++) {
			mean += (devAssignment[i] == row) * dataset[i * P + col];
			counter += (devAssignment[i] == row);
		}

		centroids[row * P + col] = mean / counter;
	}
}

__global__ void computeDistances(float* dataset, float* centroids,
		float* distances, int P, int k) {
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;
	float dist = 0.0;
	if (row < N && col < K) {
		for (int i = 0; i < P; i++) {
			dist += (dataset[row * P + i]
					- centroids[col * P + i])
					* (dataset[row * P + i]
							- centroids[col * P + i]);
		}
		distances[row * k + col] = dist;
	}
}

__global__ void computeMin(float *distances, short* devAssignment, int k,
		int n) {
	unsigned int ty = threadIdx.y;
	int by = blockIdx.y;
	unsigned int row = by * blockDim.y + ty;
	if (row < n) {
		float min = distances[row * k];
		short minInd = 0;
		for (int i = 1; i < k; i++) {
			bool conf = (min - distances[row * k + i] <= 0);
			min = conf * min + (1 - conf) * distances[row * k + i];
			minInd = conf * minInd + (1 - conf) * i;
		}
		devAssignment[row] = minInd;
	}
}

__global__ void assignPartition(short* S, int* minIndex, int n) {
	int by = blockIdx.y;
	int ty = threadIdx.y;
	int row = by * blockDim.y + ty;

	if (row < n) {
		int pos = minIndex[row];
		S[pos * n + row] = true;
	}

}

int generateRandomDataOnDevice(float *devData, int n, int P) {
	curandGenerator_t generator;
	CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, SEED));
	CURAND_CALL(curandGenerateUniform(generator, devData, n * P));
	CURAND_CALL(curandDestroyGenerator(generator));
	return 0;
}

void printFile(std::string path, std::string filename, int n, int k,
		int P, float *devDataset, short *devAssignment,
		float *devMean) {

	float *mean, *dataset;
	short *assignment;
	mean = (float*) malloc(k * P * sizeof(float));
	dataset = (float*) malloc(n * P * sizeof(float));
	assignment = (short*) malloc(n * sizeof(short));

	CUDA_CHECK_RETURN(
			cudaMemcpy(mean, devMean, P * k * sizeof(float),
					cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(
			cudaMemcpy(dataset, devDataset, n * P * sizeof(float),
					cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(
			cudaMemcpy(assignment, devAssignment, n * sizeof(short),
					cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	ofstream outputfile;
	outputfile.open(path + filename, ios::out);
	for (int i = 0; i < K; i++) {
		std::string toFile = "media" + to_string(i);
		for (int j = 0; j < P; j++) {
			toFile = toFile + " " + to_string(mean[i * P + j]);
		}
		toFile = toFile + "|";
		outputfile << toFile;
	}
	for (int i = 0; i < N; i++) {
		std::string toFile = "";
		toFile = to_string(assignment[i]);
		for (int j = 0; j < P; j++) {
			toFile = toFile + " " + to_string(dataset[i * P + j]);
		}
		toFile = toFile + "|";
		outputfile << toFile;
	}
	outputfile.close();
	free(mean);
	free(dataset);
	free(assignment);
}

bool stopCriterionSequential(short *assignment, short *oldAssignment, int len) {
	for (int i = 0; i < len; i++) {
		if (assignment[i] != oldAssignment[i]) {
			return false;
		}
	}
	return true;
}

bool stopCriterionSequential(float *vector1, float *vector2, int len) {
	for (int i = 0; i < len; i++) {
		if (abs(vector1[i] - vector2[i]) > TOL) {
			return false;
		}
	}
	return true;
}

int kMeans(float *devData, short *devAssignment, float *devMean, int n, int k,
		int P) {
	printf("KMEANS nuovo \r\n");
	dim3 DimBlockInizToZero(256, 1, 1);
	dim3 DimGridInizToZero(((2 * K) / 256) + 1, 1, 1);
	dim3 DimBlock2(n, k, 1);
	dim3 DimBlockAssignment(1, n, 1);
	dim3 DimBlockUpdate(P, k, 1);
	dim3 DimGrid(1, 1, 1);
	dim3 DimBlockSum(P, 256, 1);
	dim3 DimGridSum(1, (N / 256) + 1, 1);
	dim3 DimBlockMean(P, 256, 1);
	dim3 DimGridMean(1, (K / 256) + 1, 1);
	dim3 DimBlockDistances(4, 256, 1);
	dim3 DimGridDistances((k / 4) + 1, (n / 256) + 1, 1);
	dim3 DimBlockMin(1, 256, 1);
	dim3 DimGridMin(1, (n / 256) + 1, 1);
	dim3 DimBlockAss(1, 256, 1);
	dim3 DimGridAss(1, (n / 256) + 1, 1);
	dim3 DimGridInizPart((k / 1024) + 1, 1, 1);
	dim3 DimBlockInizPart(1024, 1, 1);
	dim3 DimGridInizMean(1, k / 16 + 1, 1);
	dim3 DimBlockInizMean(P, 16, 1);
	cudaMemset(devAssignment, 0, n * sizeof(short));
	initializeAssignment<<<DimGridInizPart, DimBlockInizPart>>>(
			devAssignment, n);
	cudaDeviceSynchronize();
	initializeMean<<<DimGridInizMean, DimBlockInizMean>>>(devMean, devData, n,
			P);
	cudaDeviceSynchronize();
	short *devOldAssignment;
	CUDA_CHECK_RETURN(
			cudaMalloc((void** )&devOldAssignment, n * sizeof(short)));
	float *hostData, *hostMean, *hostOldMean;
	hostData = (float*) malloc(n * P * sizeof(float));
	short *hostAssignment, *hostOldAssignment;
	hostAssignment = (short*) malloc(n * sizeof(short));
	hostOldAssignment = (short*) malloc(n * sizeof(short));
	int *deviceOutputVector;
	hostMean = (float*) malloc(k * P * sizeof(float));
	hostOldMean = (float*) malloc(k * P * sizeof(float));
	CUDA_CHECK_RETURN(
			cudaMalloc((void** )&deviceOutputVector, k * sizeof(int)));
	bool stopCriterion = false;
	float *devDistances;
	CUDA_CHECK_RETURN(
			cudaMalloc((void** )&devDistances, n * k * sizeof(float)));
	int* devCounter, *hostCounter;
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devCounter, k * sizeof(int)));
	hostCounter = (int*) malloc(k * sizeof(int));
	while (!stopCriterion) { 
		CUDA_CHECK_RETURN(cudaMemset(devOldAssignment, 0, n * sizeof(short)));
		CUDA_CHECK_RETURN(cudaMemset(devCounter, 0, k * sizeof(int)));
		cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(
				cudaMemcpy(devOldAssignment, devAssignment, n * sizeof(short),
						cudaMemcpyDeviceToDevice));
		cudaDeviceSynchronize();
		computeDistances<<<DimGridDistances, DimBlockDistances>>>(devData,
				devMean, devDistances, P, k);
		cudaDeviceSynchronize();
		computeMin<<<DimGridMin, DimBlockMin>>>(devDistances, devAssignment, k,
				n);
		cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(
				cudaMemcpy(hostOldMean, devMean, P * k * sizeof(float),
						cudaMemcpyDeviceToHost));
		initializeVectorToValue<<<DimGridInizToZero, DimBlockInizToZero>>>(
				devMean, 0.0, P * K);
		cudaDeviceSynchronize();
		computeSum37<<<DimGridSum, DimBlockSum>>>(devData, devAssignment,
				devMean, devCounter);
		cudaDeviceSynchronize();
		computeMean37<<<DimGridMean, DimBlockMean>>>(devMean, devCounter);
		cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(
				cudaMemcpy(hostMean, devMean, P * k * sizeof(float),
						cudaMemcpyDeviceToHost));
		stopCriterion = stopCriterionSequential(hostOldMean, hostMean,
		K * P);
		/*
		 printFile("/home/cecca/Desktop/RemoteCompilation/TXTDATA/",
		 to_string(iter) + ".txt", n, k, P, devData,
		 devAssignment, devMean);
		*/
	}
	return 0;
}


void printFileForSeq(std::string path, std::string filename, float *dataset,
		short *assignment, float *mean) {
	ofstream outputfile;
	outputfile.open(path + filename, ios::out);
	for (int i = 0; i < K; i++) {
		std::string toFile = "media" + to_string(i);
		for (int j = 0; j < P; j++) {
			toFile = toFile + " " + to_string(mean[i * P + j]);

		}
		toFile = toFile + "|";
		outputfile << toFile;
	}
	for (int i = 0; i < N; i++) {
		std::string toFile = "";
		toFile = to_string(assignment[i]);
		for (int j = 0; j < P; j++) {
			toFile = toFile + " " + to_string(dataset[i * P + j]);
		}
		toFile = toFile + "|";
		outputfile << toFile;
	}
	outputfile.close();
}


int main(int argc, char *argv[]) {
	system("rm -rf /home/cecca/Desktop/RemoteCompilation/TXTDATA");
	system("mkdir /home/cecca/Desktop/RemoteCompilation/TXTDATA");
	size_t n;
	size_t i;
	int k;
	int P = 2;
	if (argc > 1) {
		n = atoi(argv[1]);
	} else {
		n = N;
	}
	if (argc > 2) {
		k = atoi(argv[2]);
	} else {
		k = K;
	}

	float *devCentroids, *hostCentroids;
	int sizeOfData = P * n * sizeof(float);
	short *hostAssignment, *devAssignment;
	float *devData, *hostData;
	float *devMean, *hostMean;
	hostData = (float*) malloc(sizeOfData);
	hostCentroids = (float*) malloc((sizeOfData * k) / n);
	hostMean = (float*) malloc((sizeOfData * k) / n);
	hostAssignment = (short*) malloc(n * sizeof(short));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devData, sizeOfData));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devMean, (sizeOfData * k) / n));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devAssignment, n * sizeof(short)));
	CUDA_CHECK_RETURN(cudaMalloc((void** )&devCentroids, (sizeOfData * k) / n));
	generateRandomDataOnDevice(devData, n, P);
//###############################
	clock_t start, end;
	double cpu_time_used;
	start = clock();
	kMeans(devData, devAssignment, devCentroids, n, k, P);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("TEMPO: %f SECONDI \r\n", cpu_time_used);
//###############################
	CUDA_CHECK_RETURN(cudaFree(devAssignment));
	CUDA_CHECK_RETURN(cudaFree(devMean));
	CUDA_CHECK_RETURN(cudaFree(devData));
	free(hostData);
	free(hostMean);
	free(hostAssignment);

	return 0;

}

static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

