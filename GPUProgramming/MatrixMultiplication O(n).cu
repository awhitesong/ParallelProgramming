#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<stdlib.h>
#include<assert.h>
using namespace std;
void initCUDA(int*, int*, int*, int);

__global__ void multiply_(int *c, int *a, int *b, int n){
	size_t row, col;
	int k,temp=0;
	row = threadIdx.x;
	col = threadIdx.y;
	for (k = 0; k < n; k++){
		temp += a[row*n + k] * b[k*n + col];
	}
	c[row*n + col] = temp;
}

main(){
	int i, j, n, *a, *b, *c, k, *gpu_result;
	cout << "Enter the square matrix's dimensions (maximum 32)\n";
	cin >> n;
	assert(n <= 32);
	a = (int*)malloc(sizeof(int)*n*n);
	b = (int*)malloc(sizeof(int)*n*n);
	gpu_result = (int*)malloc(sizeof(int)*n*n);
	c = (int*)calloc(n*n, sizeof(int));
	cout << "Enter Matrice A: \n\n";
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			cin >> a[i*n + j];
		}
	}
	cout << "\nEnter Matrice B: \n\n";
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			cin >> b[i*n + j];
		}
	}
	cout << "\nCPU Multiplication in O(n^3):\n\n";
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			for (k = 0; k < n; k++){
				c[i*n + j] += a[i*n + k] * b[k*n + j];
			}
		}
	}
	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			cout << c[i*n + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	initCUDA(a, b, gpu_result, n);

	cout << "GPU Multiplication in O(n):\n\n";

	for (i = 0; i < n; i++){
		for (j = 0; j < n; j++){
			cout << gpu_result[i*n + j] << " ";
		}
		cout << endl;
	}

	free(a);
	free(b);
	free(c);
	free(gpu_result);
}


void initCUDA(int *a, int *b, int *result, int n){
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, n*n*sizeof(int));
	cudaMalloc(&d_b, n*n*sizeof(int));
	cudaMalloc(&d_c, n*n*sizeof(int));
	cudaMemcpy(d_a, a, n*n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n*n*sizeof(int), cudaMemcpyHostToDevice);
	dim3 threadsinablock(n, n);
	dim3 blocksingrid(1, 1);
	multiply_ <<<blocksingrid, threadsinablock >>>(d_c, d_a, d_b, n);
	cudaMemcpy(result, d_c, n*n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

}