#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>

//CUDA RunTime API
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define TILE_WIDTH 32
#define MAXVALUE 1e8

#define checkCudaErrors(call)                                                                   \
{                                                                                               \
    cudaError_t cudaStatus = call;                                                              \
    if (cudaStatus != cudaSuccess)                                                              \
    {                                                                                           \
        std::cerr << "CUDA API error: " << cudaGetErrorString(cudaStatus) << " at "            \
                  << __FILE__ << " line " << __LINE__ << "." << std::endl;                      \
        exit(EXIT_FAILURE);                                                                     \
    }                                                                                           \
}

__global__ void fw_kernel1(float *d_Len, int *d_Path, const unsigned int dim, unsigned int sub_dim)
{
	__shared__ float sh_Len[TILE_WIDTH * TILE_WIDTH];
	__shared__ int sh_Path[TILE_WIDTH * TILE_WIDTH];

	int i = threadIdx.y;
	int j = threadIdx.x;

	int sub_i = threadIdx.y;
	int sub_j = threadIdx.x;

	int sh_num = sub_i * TILE_WIDTH + sub_j;
	int d_num = i * dim + j;
	int sub_ik, sub_kj;
	if (i < dim && j < dim)
	{
		sh_Len[sh_num] = d_Len[d_num];
		sh_Path[sh_num] = d_Path[d_num];
	}
	else{
		sh_Len[sh_num] = MAXVALUE;
		sh_Path[sh_num] = -1;
	}
	__syncthreads();

	for (int sub_k = 0; sub_k < sub_dim; sub_k++)
	{
		sub_ik = sub_i * TILE_WIDTH + sub_k;
		sub_kj = sub_k * TILE_WIDTH + sub_j;
		if (sh_Len[sh_num] > sh_Len[sub_ik] + sh_Len[sub_kj])
		{
			sh_Len[sh_num] = sh_Len[sub_ik] + sh_Len[sub_kj];
			sh_Path[sh_num] = sh_Path[sub_kj];
		}
		__syncthreads();
	}

	if (i < dim && j < dim)
	{
		d_Len[d_num] = sh_Len[sh_num];
		d_Path[d_num] = sh_Path[sh_num];
	}
}

__global__ void FW_kernel1A2(int B, float *d_Len, int *d_Path, const unsigned int dim, unsigned int sub_dim)
{

	__shared__ float c_sh_Len[TILE_WIDTH * TILE_WIDTH];
	__shared__ int c_sh_Path[TILE_WIDTH * TILE_WIDTH];
	__shared__ float p_sh_Len[TILE_WIDTH * TILE_WIDTH];
	__shared__ int p_sh_Path[TILE_WIDTH * TILE_WIDTH];

	//diag Matrix
	int p_i = B * TILE_WIDTH + threadIdx.y;
	int p_j = B * TILE_WIDTH + threadIdx.x;

	int skipCenterBlock = min((blockIdx.x + 1) / (B + 1), 1);
	int c_i, c_j;

	if (blockIdx.y == 0)
	{
		// B matrix
		c_i = p_i;
		c_j = (blockIdx.x + skipCenterBlock) * TILE_WIDTH + threadIdx.x;
	}
	else
	{
		// A matirx
		c_i = (blockIdx.x + skipCenterBlock) * TILE_WIDTH + threadIdx.y;
		c_j = p_j;
	}

	int sub_i = threadIdx.y;
	int sub_j = threadIdx.x;

	int sh_num = sub_i * TILE_WIDTH + sub_j;
	int d_c_num = c_i * dim + c_j;
	int d_p_num = p_i * dim + p_j;
	int sub_ik, sub_kj;

	if (p_i < dim && p_j < dim)
	{
		p_sh_Len[sh_num] = d_Len[d_p_num];
		p_sh_Path[sh_num] = d_Path[d_p_num];
	}
	else{
		p_sh_Len[sh_num] = MAXVALUE;
		p_sh_Path[sh_num] = -1;
	}
	if (c_i < dim && c_j < dim)
	{
		c_sh_Len[sh_num] = d_Len[d_c_num];
		c_sh_Path[sh_num] = d_Path[d_c_num];
	}
	else{
		c_sh_Len[sh_num] = MAXVALUE;
		c_sh_Path[sh_num] = -1;
	}
	__syncthreads();

	//update the diag matrix
	for (int sub_k = 0; sub_k < sub_dim; sub_k++)
	{

		sub_ik = sub_i * TILE_WIDTH + sub_k;
		sub_kj = sub_k * TILE_WIDTH + sub_j;

		if (p_sh_Len[sh_num] > p_sh_Len[sub_ik] + p_sh_Len[sub_kj])
		{
			p_sh_Len[sh_num] = p_sh_Len[sub_ik] + p_sh_Len[sub_kj];
			p_sh_Path[sh_num] = p_sh_Path[sub_kj];
		}
		__syncthreads();
	}

	for (int sub_k = 0; sub_k < sub_dim; sub_k++)
	{
		sub_ik = sub_i * TILE_WIDTH + sub_k;
		sub_kj = sub_k * TILE_WIDTH + sub_j;
		if (blockIdx.y == 1)
		{
			//A matrix
			if (c_sh_Len[sh_num] > c_sh_Len[sub_ik] + p_sh_Len[sub_kj])
			{
				c_sh_Len[sh_num] = c_sh_Len[sub_ik] + p_sh_Len[sub_kj];
				c_sh_Path[sh_num] = p_sh_Path[sub_kj];
			}
		}
		if (blockIdx.y == 0)
		{
			//B matrix
			if (c_sh_Len[sh_num] > p_sh_Len[sub_ik] + c_sh_Len[sub_kj])
			{
				c_sh_Len[sh_num] = p_sh_Len[sub_ik] + c_sh_Len[sub_kj];
				c_sh_Path[sh_num] = c_sh_Path[sub_kj];
			}
		}
		__syncthreads();
	}

	if (blockIdx.y == 0 && blockIdx.x == 0)
	{
		// diag matrix
		if (p_i < dim && p_j < dim)
		{
			d_Len[d_p_num] = p_sh_Len[sh_num];
			d_Path[d_p_num] = p_sh_Path[sh_num];
		}
		if (c_i < dim && c_j < dim)
		{
			d_Len[d_c_num] = c_sh_Len[sh_num];
			d_Path[d_c_num] = c_sh_Path[sh_num];
		}
	}
	else
	{
		//TODO
		if (c_i < dim && c_j < dim)
		{
			d_Len[d_c_num] = c_sh_Len[sh_num];
			d_Path[d_c_num] = c_sh_Path[sh_num];
		}
	}
}

__global__ void fw_kernel3(int B, float *d_Len, int *d_Path, const unsigned int dim, unsigned int sub_dim)
{
	__shared__ float c_sh_Len[TILE_WIDTH * TILE_WIDTH];
	__shared__ int c_sh_Path[TILE_WIDTH * TILE_WIDTH];
	__shared__ float p1_sh_Len[TILE_WIDTH * TILE_WIDTH];
	__shared__ float p2_sh_Len[TILE_WIDTH * TILE_WIDTH];
	__shared__ int p2_sh_Path[TILE_WIDTH * TILE_WIDTH];

	int skipCenterBlockX = min((blockIdx.x + 1) / (B + 1), 1);
	int skipCenterBlockY = min((blockIdx.y + 1) / (B + 1), 1);

	int c_i = (blockIdx.y + skipCenterBlockY) * TILE_WIDTH + threadIdx.y;
	int c_j = (blockIdx.x + skipCenterBlockX) * TILE_WIDTH + threadIdx.x;
	int p1_i = c_i;
	int p1_j = B * TILE_WIDTH + threadIdx.x;
	int p2_i = B * TILE_WIDTH + threadIdx.y;
	int p2_j = c_j;

	int sub_i = threadIdx.y;
	int sub_j = threadIdx.x;

	int sh_num = sub_i * TILE_WIDTH + sub_j;
	int d_c_num = c_i * dim + c_j;
	int d_p1_num = p1_i * dim + p1_j;
	int d_p2_num = p2_i * dim + p2_j;

	if (p1_i < dim && p1_j < dim)
	{
		p1_sh_Len[sh_num] = d_Len[d_p1_num];
	}
	else{
		p1_sh_Len[sh_num] = MAXVALUE;
	}
	if (p2_i < dim && p2_j < dim)
	{
		p2_sh_Len[sh_num] = d_Len[d_p2_num];
		p2_sh_Path[sh_num] = d_Path[d_p2_num];
	}
	else{
		p2_sh_Len[sh_num] = MAXVALUE;
		p2_sh_Path[sh_num] = -1;
	}
	if (c_i < dim && c_j < dim)
	{
		c_sh_Len[sh_num] = d_Len[d_c_num];
		c_sh_Path[sh_num] = d_Path[d_c_num];
	}
	else{
		c_sh_Len[sh_num] = MAXVALUE;
		c_sh_Path[sh_num] = -1;
	}

	__syncthreads();

	int sub_ik, sub_kj;
	float sh_len;

	for (int sub_k = 0; sub_k < sub_dim; sub_k++)
	{
		sub_ik = sub_i * TILE_WIDTH + sub_k;
		sub_kj = sub_k * TILE_WIDTH + sub_j;
		sh_len = p1_sh_Len[sub_ik] + p2_sh_Len[sub_kj];
		if (c_sh_Len[sh_num] > sh_len)
		{
			c_sh_Len[sh_num] = sh_len;
			c_sh_Path[sh_num] = p2_sh_Path[sub_kj];
		}
	}

	__syncthreads();

	if (c_i < dim && c_j < dim)
	{
		d_Len[d_c_num] = c_sh_Len[sh_num];
		d_Path[d_c_num] = c_sh_Path[sh_num];
	}
}

__global__ void floyd_baseline(const int k, float *dist, int *path, const int dim)
{
	int i = TILE_WIDTH * blockIdx.y + threadIdx.y;
	int j = TILE_WIDTH * blockIdx.x + threadIdx.x;

	if (i < dim && j < dim)
	{
		if (dist[i * dim + j] > dist[i * dim + k] + dist[k * dim + j])
		{
			dist[i * dim + j] = dist[i * dim + k] + dist[k * dim + j];
			path[i * dim + j] = path[k * dim + j];
		}
	}
}

void floyd_GPU_Nvidia_path(int num_node, float *arc, int *path_node)
{
	unsigned int n = num_node * num_node;
	float *d_Len;
	int *d_Path;

	checkCudaErrors(cudaMalloc(&d_Len, n * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_Path, n * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_Len, arc, n * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Path, path_node, n * sizeof(int), cudaMemcpyHostToDevice));

	const unsigned int numberOfBlocks = ceil((float)num_node / (float)TILE_WIDTH);
	unsigned int sub_dim;

#ifdef PROFILER
	cudaEvent_t start, stop;   //declare
	cudaEventCreate(&start);   //set up
	cudaEventCreate(&stop);	   //set up
	cudaEventRecord(start, 0); //start
#endif

	if (numberOfBlocks == 1)
	{
		dim3 Grid_square(1, 1);
		dim3 Block_square(TILE_WIDTH, TILE_WIDTH);
		fw_kernel1<<<Grid_square, Block_square>>>(d_Len, d_Path, num_node, num_node);
	}
	else
	{
		for (int B = 0; B < numberOfBlocks; B++)
		{
			if (B == (numberOfBlocks - 1) && num_node % TILE_WIDTH != 0)
			{
				sub_dim = num_node % TILE_WIDTH;
			}
			else
			{
				sub_dim = TILE_WIDTH;
			}
			FW_kernel1A2<<<dim3(numberOfBlocks - 1, 2), dim3(TILE_WIDTH, TILE_WIDTH)>>>(B, d_Len, d_Path, num_node, sub_dim);
			fw_kernel3<<<dim3(numberOfBlocks - 1, numberOfBlocks - 1), dim3(TILE_WIDTH, TILE_WIDTH)>>>(B, d_Len, d_Path, num_node, sub_dim);
		}
	}

	// for (int i = 0; i < num_node; i++)
	// {
	// 	dim3 Grid_square(num_node / TILE_WIDTH + 1, num_node / TILE_WIDTH + 1);
	// 	dim3 Block_square(TILE_WIDTH, TILE_WIDTH);
	// 	floyd_baseline<<<Grid_square, Block_square>>>(i, d_Len, d_Path, num_node);
	// }

#ifdef PROFILER
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float eTime;
	cudaEventElapsedTime(&eTime, start, stop);
	printf("the floyd flops is: %f\n", (float)num_node * num_node * num_node * 2.0 * 1000.0 / eTime);
#endif

	checkCudaErrors(cudaMemcpy(arc, d_Len, n * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(path_node, d_Path, n * sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(d_Len);
	cudaFree(d_Path);
}

__global__ void floyd_path_A_kernel(float *mat, int *mat_path, float *A, int *A_path, int row, int col)
{
	// A coordinate
	int c_i = blockDim.y * blockIdx.y + threadIdx.y;
	int c_j = blockDim.x * blockIdx.x + threadIdx.x;

	if (c_i < row && c_j < col)
	{
		float tmp = MAXVALUE;
		int tmp_path = -1;
		for (int k = 0; k < col; k++)
		{
			if (tmp > A[c_i * col + k] + mat[k * col + c_j])
			{
				tmp = A[c_i * col + k] + mat[k * col + c_j];
				tmp_path = mat_path[k * col + c_j];
			}
		}
		__syncthreads();
		A[c_i * col + c_j] = tmp;
		A_path[c_i * col + c_j] = tmp_path;
	}
}

void floyd_path_A_Nvidia(float *A,int *A_path,const int row,const int col,float *diag,int *diag_path)
{
	float *d_diag;
	checkCudaErrors(cudaMalloc(&d_diag, col * col * sizeof(float)));
	int *d_diag_path;
	checkCudaErrors(cudaMalloc(&d_diag_path, col * col * sizeof(int)));
	float *d_A;
	checkCudaErrors(cudaMalloc(&d_A, row * col * sizeof(float)));
	int *d_A_path;
	checkCudaErrors(cudaMalloc(&d_A_path, row * col * sizeof(int)));

	checkCudaErrors(cudaMemcpy(d_diag, diag, col * col * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_A, A, row * col * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_diag_path, diag_path, col * col * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_A_path, A_path, row * col * sizeof(int), cudaMemcpyHostToDevice));

	floyd_path_A_kernel<<<dim3(col / TILE_WIDTH + 1, row / TILE_WIDTH + 1, 1), dim3(TILE_WIDTH, TILE_WIDTH, 1)>>>(d_diag, d_diag_path, d_A, d_A_path, row, col);

	checkCudaErrors(cudaMemcpy(A_path, d_A_path, row * col * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(A, d_A, row * col * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_diag);
	cudaFree(d_diag_path);
	cudaFree(d_A);
	cudaFree(d_A_path);
}

__global__ void floyd_path_B_kernel(float *mat, int *mat_path, float *B, int *B_path, int row, int col)
{
	// B coordinate
	int c_i = blockDim.y * blockIdx.y + threadIdx.y;
	int c_j = blockDim.x * blockIdx.x + threadIdx.x;

	if (c_i < row && c_j < col)
	{
		float tmp = MAXVALUE;
		int tmp_path = -1;
		for (int k = 0; k < row; k++)
		{
			if (tmp > mat[c_i * row + k] + B[k * col + c_j])
			{
				tmp = mat[c_i * row + k] + B[k * col + c_j];
				tmp_path = B_path[k * col + c_j];
			}
		}
		__syncthreads();
		B[c_i * col + c_j] = tmp;
		B_path[c_i * col + c_j] = tmp_path;
	}
}

void floyd_path_B_Nvidia(float *B,int *B_path,const int row,const int col,float *diag,int *diag_path)
{
	float *d_diag;
	checkCudaErrors(cudaMalloc(&d_diag, row * row * sizeof(float)));
	int *d_diag_path;
	checkCudaErrors(cudaMalloc(&d_diag_path, row * row * sizeof(int)));
	float *d_B;
	checkCudaErrors(cudaMalloc(&d_B, row * col * sizeof(float)));
	int *d_B_path;
	checkCudaErrors(cudaMalloc(&d_B_path, row * col * sizeof(int)));

	checkCudaErrors(cudaMemcpy(d_diag, diag, row * row * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, row * col * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_diag_path, diag_path, row * row * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B_path, B_path, row * col * sizeof(int), cudaMemcpyHostToDevice));

	floyd_path_B_kernel<<<dim3(col / TILE_WIDTH + 1, row / TILE_WIDTH + 1, 1), dim3(TILE_WIDTH, TILE_WIDTH, 1)>>>(d_diag, d_diag_path, d_B, d_B_path, row, col);

	checkCudaErrors(cudaMemcpy(B_path, d_B_path, row * col * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(B, d_B, row * col * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(d_diag);
	cudaFree(d_diag_path);
	cudaFree(d_B);
	cudaFree(d_B_path);
}

__global__ void minplus_kernel_path(float *A, float *B, int *B_path, float *C, int *C_path, int m, int n, int k)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B_path[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

	float Cvalue = MAXVALUE;
    int Cpath = -1;
	
	if ((row < m) && (col < n)){
		Cvalue = C[row * n + col];
		Cpath = C_path[row * n + col];
	}
    

    for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
    {
        if (row < m && t * TILE_WIDTH + tx < k){
			ds_A[tx][ty] = A[row * k + t * TILE_WIDTH + tx];
		}
        else{
			ds_A[tx][ty] = MAXVALUE;
		}

        if (t * TILE_WIDTH + ty < k && col < n){
            ds_B[tx][ty] = B[(t * TILE_WIDTH + ty) * n + col];
            ds_B_path[tx][ty] = B_path[(t * TILE_WIDTH + ty) * n + col];
        }
        else{
            ds_B[tx][ty] = MAXVALUE;
            ds_B_path[tx][ty] = -1;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
        {
            if (Cvalue > ds_A[i][ty] + ds_B[tx][i])
            {
                Cvalue = ds_A[i][ty] + ds_B[tx][i];
                Cpath = ds_B_path[tx][i];
            }
        }
        __syncthreads();
    }

	if ((row < m) && (col < n))
	{
		C[row * n + col] = Cvalue;
		C_path[row * n + col] = Cpath;
	}
}

void floyd_min_plus_Nvidia(float *mat1, float *mat2, int *mat2_path,
    						float *res, int *res_path, int m, int n, int k)
{
	float *d_a;
    float *d_b;
    float *d_c;
    int *d_b_path;
    int *d_c_path;
    checkCudaErrors(cudaMalloc(&d_a, m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b, k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c, m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b_path, k * n * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_c_path, m * n * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_a, mat1, m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, mat2, k * n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b_path, mat2_path, k * n * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c, res, m * n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_c_path, res_path, m * n * sizeof(int), cudaMemcpyHostToDevice));

    const unsigned int GridDim_x = n / TILE_WIDTH + 1;
    const unsigned int GridDim_y = m / TILE_WIDTH + 1;

    minplus_kernel_path<<<dim3(GridDim_x, GridDim_y), dim3(TILE_WIDTH, TILE_WIDTH)>>>(d_a, d_b, d_b_path, d_c, d_c_path, m, n, k);

    checkCudaErrors(cudaMemcpy(res, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(res_path, d_c_path, m * n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_b_path);
    cudaFree(d_c_path);
}