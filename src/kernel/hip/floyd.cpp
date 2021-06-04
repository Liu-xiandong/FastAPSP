#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "parameter.h"

//CUDA RunTime API
#include "hip/hip_runtime.h"
#include <hc_defines.h>

#define TILE_WIDTH 32

#define CHECK(cmd)                                                                         \
	{                                                                                      \
		hipError_t error = cmd;                                                            \
		if (error != hipSuccess)                                                           \
		{                                                                                  \
			fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, \
					__FILE__, __LINE__);                                                   \
			exit(EXIT_FAILURE);                                                            \
		}                                                                                  \
	}

__global__ void fw_kernel1(float *d_Len, int *d_Path, const unsigned int dim, unsigned int sub_dim)
{
	__shared__ float sh_Len[TILE_WIDTH * TILE_WIDTH];
	__shared__ int sh_Path[TILE_WIDTH * TILE_WIDTH];

	int i = hipThreadIdx_y;
	int j = hipThreadIdx_x;

	int sub_i = hipThreadIdx_y;
	int sub_j = hipThreadIdx_x;

	int sh_num = sub_i * TILE_WIDTH + sub_j;
	int d_num = i * dim + j;
	int sub_ik, sub_kj;
	if (i < dim && j < dim)
	{
		sh_Len[sh_num] = d_Len[d_num];
		sh_Path[sh_num] = d_Path[d_num];
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

	__shared__ float c_sh_Len[TILE_WIDTH * TILE_WIDTH]; //共行或共列距离子矩阵;
	__shared__ int c_sh_Path[TILE_WIDTH * TILE_WIDTH];	//共行或共列路径子矩阵;
	__shared__ float p_sh_Len[TILE_WIDTH * TILE_WIDTH]; //对角线距离子矩阵；
	__shared__ int p_sh_Path[TILE_WIDTH * TILE_WIDTH];	//对角线路径子矩阵；

	int p_i = B * TILE_WIDTH + threadIdx.y;
	int p_j = B * TILE_WIDTH + threadIdx.x;

	int skipCenterBlock = min((blockIdx.x + 1) / (B + 1), 1);
	int c_i, c_j;

	if (blockIdx.y == 0)
	{
		c_i = p_i;
		c_j = (blockIdx.x + skipCenterBlock) * TILE_WIDTH + threadIdx.x;
	}
	else
	{
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
	if (c_i < dim && c_j < dim)
	{
		c_sh_Len[sh_num] = d_Len[d_c_num];
		c_sh_Path[sh_num] = d_Path[d_c_num];
	}
	__syncthreads();

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
			if (c_sh_Len[sh_num] > c_sh_Len[sub_ik] + p_sh_Len[sub_kj])
			{
				c_sh_Len[sh_num] = c_sh_Len[sub_ik] + p_sh_Len[sub_kj];
				c_sh_Path[sh_num] = p_sh_Path[sub_kj];
			}
		}
		if (blockIdx.y == 0)
		{
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
	__shared__ int p1_sh_Path[TILE_WIDTH * TILE_WIDTH];
	__shared__ float p2_sh_Len[TILE_WIDTH * TILE_WIDTH];
	__shared__ int p2_sh_Path[TILE_WIDTH * TILE_WIDTH];

	int skipCenterBlockX = min((hipBlockIdx_x + 1) / (B + 1), 1);
	int skipCenterBlockY = min((hipBlockIdx_y + 1) / (B + 1), 1);

	int c_i = (hipBlockIdx_y + skipCenterBlockY) * TILE_WIDTH + hipThreadIdx_y;
	int c_j = (hipBlockIdx_x + skipCenterBlockX) * TILE_WIDTH + hipThreadIdx_x;
	int p1_i = c_i;
	int p1_j = B * TILE_WIDTH + hipThreadIdx_x;
	int p2_i = B * TILE_WIDTH + hipThreadIdx_y;
	int p2_j = c_j;

	int sub_i = hipThreadIdx_y;
	int sub_j = hipThreadIdx_x;

	int sh_num = sub_i * TILE_WIDTH + sub_j;
	int d_c_num = c_i * dim + c_j;
	int d_p1_num = p1_i * dim + p1_j;
	int d_p2_num = p2_i * dim + p2_j;

	if (p1_i < dim && p1_j < dim)
	{
		p1_sh_Len[sh_num] = d_Len[d_p1_num];
		p1_sh_Path[sh_num] = d_Path[d_p1_num];
	}
	if (p2_i < dim && p2_j < dim)
	{
		p2_sh_Len[sh_num] = d_Len[d_p2_num];
		p2_sh_Path[sh_num] = d_Path[d_p2_num];
	}
	if (c_i < dim && c_j < dim)
	{
		c_sh_Len[sh_num] = d_Len[d_c_num];
		c_sh_Path[sh_num] = d_Path[d_c_num];
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

extern "C" void floyd_AMD_path(int num_node, float *arc, int *path_node)
{
	unsigned int n = num_node * num_node;
	struct timeval begin, end;
	float *d_Len;
	int *d_Path;
	CHECK(hipMalloc(&d_Len, n * sizeof(float)));
	CHECK(hipMalloc(&d_Path, n * sizeof(int)));

	CHECK(hipMemcpy(d_Len, arc, n * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_Path, path_node, n * sizeof(int), hipMemcpyHostToDevice));

	const unsigned int numberOfBlocks = ceil((float)num_node / (float)TILE_WIDTH);
	unsigned int sub_dim;

	if (numberOfBlocks == 1)
	{
		hipLaunchKernelGGL(fw_kernel1, dim3(1, 1), dim3(TILE_WIDTH, TILE_WIDTH), 0, 0, d_Len, d_Path, num_node, num_node);
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
			hipLaunchKernelGGL(FW_kernel1A2, dim3(numberOfBlocks - 1, 2), dim3(TILE_WIDTH, TILE_WIDTH), 0, 0, B, d_Len, d_Path, num_node, sub_dim);
			hipLaunchKernelGGL(fw_kernel3, dim3(numberOfBlocks - 1, numberOfBlocks - 1), dim3(TILE_WIDTH, TILE_WIDTH), 0, 0, B, d_Len, d_Path, num_node, sub_dim);
		}
	}
	hipDeviceSynchronize();

	CHECK(hipMemcpy(arc, d_Len, n * sizeof(float), hipMemcpyDeviceToHost));
	CHECK(hipMemcpy(path_node, d_Path, n * sizeof(int), hipMemcpyDeviceToHost));

	hipFree(d_Len);
	hipFree(d_Path);
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

extern "C" void floyd_path_A_AMD(float *A, int *A_path, const int row, const int col, float *diag, int *diag_path)
{
	float *d_diag;
	CHECK(hipMalloc(&d_diag, col * col * sizeof(float)));
	int *d_diag_path;
	CHECK(hipMalloc(&d_diag_path, col * col * sizeof(int)));
	float *d_A;
	CHECK(hipMalloc(&d_A, row * col * sizeof(float)));
	int *d_A_path;
	CHECK(hipMalloc(&d_A_path, row * col * sizeof(int)));

	CHECK(hipMemcpy(d_diag, diag, col * col * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_A, A, row * col * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_diag_path, diag_path, col * col * sizeof(int), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_A_path, A_path, row * col * sizeof(int), hipMemcpyHostToDevice));

	hipLaunchKernelGGL(floyd_path_A_kernel, dim3(col / TILE_WIDTH + 1, row / TILE_WIDTH + 1), dim3(TILE_WIDTH, TILE_WIDTH), 0, 0, d_diag, d_diag_path, d_A, d_A_path, row, col);

	CHECK(hipMemcpy(A_path, d_A_path, row * col * sizeof(int), hipMemcpyDeviceToHost));
	CHECK(hipMemcpy(A, d_A, row * col * sizeof(float), hipMemcpyDeviceToHost));

	hipFree(d_diag);
	hipFree(d_diag_path);
	hipFree(d_A);
	hipFree(d_A_path);
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

extern "C" void floyd_path_B_AMD(float *B, int *B_path, const int row, const int col, float *diag, int *diag_path)
{
	float *d_diag;
	CHECK(hipMalloc(&d_diag, row * row * sizeof(float)));
	int *d_diag_path;
	CHECK(hipMalloc(&d_diag_path, row * row * sizeof(int)));
	float *d_B;
	CHECK(hipMalloc(&d_B, row * col * sizeof(float)));
	int *d_B_path;
	CHECK(hipMalloc(&d_B_path, row * col * sizeof(int)));

	CHECK(hipMemcpy(d_diag, diag, row * row * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_B, B, row * col * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_diag_path, diag_path, row * row * sizeof(int), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_B_path, B_path, row * col * sizeof(int), hipMemcpyHostToDevice));

	hipLaunchKernelGGL(floyd_path_B_kernel, dim3(col / TILE_WIDTH + 1, row / TILE_WIDTH + 1), dim3(TILE_WIDTH, TILE_WIDTH), 0, 0, d_diag, d_diag_path, d_B, d_B_path, row, col);

	CHECK(hipMemcpy(B_path, d_B_path, row * col * sizeof(int), hipMemcpyDeviceToHost));
	CHECK(hipMemcpy(B, d_B, row * col * sizeof(float), hipMemcpyDeviceToHost));

	hipFree(d_diag);
	hipFree(d_diag_path);
	hipFree(d_B);
	hipFree(d_B_path);
}

__global__ void minplus_kernel_path(float *A, float *B, int *B_path, float *C, int *C_path, unsigned int m, unsigned int n, unsigned int k)
{
	//申请共享内存，存在于每个block中
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH + 1];
	__shared__ int ds_B_path[TILE_WIDTH][TILE_WIDTH + 1];

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	float Cvalue = MAXVALUE;
	int Cpath = -1;

	if ((row < m) && (col < n))
	{
		Cvalue = C[row * n + col];
		Cpath = C_path[row * n + col];
	}

	float last_value = Cvalue;

	for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
	{
		if (row < m && t * TILE_WIDTH + tx < k)
		{
			ds_A[tx][ty] = A[row * k + t * TILE_WIDTH + tx];
		}
		else
		{
			ds_A[tx][ty] = MAXVALUE;
		}

		if (t * TILE_WIDTH + ty < k && col < n)
		{
			ds_B[tx][ty] = B[(t * TILE_WIDTH + ty) * n + col];
			ds_B_path[tx][ty] = B_path[(t * TILE_WIDTH + ty) * n + col];
		}
		else
		{
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

	if ((row < m) && (col < n) && (Cvalue < last_value))
	{
		C[row * n + col] = Cvalue;
		C_path[row * n + col] = Cpath;
	}
}

extern "C" void floyd_min_plus_AMD(float *mat1, float *mat2, int *mat2_path,
								   float *res, int *res_path, int tm, int tn, int tk)
{
	float *d_a;
	float *d_b;
	float *d_c;
	int *d_b_path;
	int *d_c_path;

	unsigned int m = static_cast<unsigned int>(tm);
	unsigned int n = static_cast<unsigned int>(tn);
	unsigned int k = static_cast<unsigned int>(tk);

	unsigned int a_num = m * k;
	unsigned int b_num = k * n;
	unsigned int c_num = m * n;

	CHECK(hipMalloc(&d_a, a_num * sizeof(float)));
	CHECK(hipMalloc(&d_b, b_num * sizeof(float)));
	CHECK(hipMalloc(&d_c, c_num * sizeof(float)));
	CHECK(hipMalloc(&d_b_path, b_num * sizeof(int)));
	CHECK(hipMalloc(&d_c_path, c_num * sizeof(int)));

	CHECK(hipMemcpy(d_a, mat1, a_num * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_b, mat2, b_num * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_b_path, mat2_path, b_num * sizeof(int), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_c, res, c_num * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_c_path, res_path, c_num * sizeof(int), hipMemcpyHostToDevice));

	const unsigned int GridDim_x = n / TILE_WIDTH + 1;
	const unsigned int GridDim_y = m / TILE_WIDTH + 1;

	hipLaunchKernelGGL(minplus_kernel_path, dim3(GridDim_x, GridDim_y), dim3(TILE_WIDTH, TILE_WIDTH), 0, 0, d_a, d_b, d_b_path, d_c, d_c_path, m, n, k);

	CHECK(hipMemcpy(res, d_c, c_num * sizeof(float), hipMemcpyDeviceToHost));
	CHECK(hipMemcpy(res_path, d_c_path, c_num * sizeof(int), hipMemcpyDeviceToHost));

	hipFree(d_a);
	hipFree(d_b);
	hipFree(d_c);
	hipFree(d_b_path);
	hipFree(d_c_path);
}

extern "C" void floyd_minplus_partition_AMD(float *mat1, float *mat2, int *mat2_path,
											float *res, int *res_path, int m, int n, int k, const int part_num)
{
	assert(part_num > 1);
	int block_size = m / part_num;
	int last_size = m - block_size * (part_num - 1);
	assert(last_size < 0);

	float *d_b;
	int *d_b_path;
	CHECK(hipMalloc(&d_b, (long long)k * n * sizeof(float)));
	CHECK(hipMalloc(&d_b_path, (long long)k * n * sizeof(int)));

	CHECK(hipMemcpy(d_b, mat2, (long long)k * n * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_b_path, mat2_path, (long long)k * n * sizeof(int), hipMemcpyHostToDevice));

	int max_size = std::max(block_size, last_size);
	float *d_a;
	float *d_c;
	int *d_c_path;
	CHECK(hipMalloc(&d_a, (long long)max_size * k * sizeof(float)));
	CHECK(hipMalloc(&d_c, (long long)max_size * n * sizeof(float)));
	CHECK(hipMalloc(&d_c_path, (long long)max_size * n * sizeof(int)));

	for (int i = 0; i < part_num; i++)
	{
		int now_size;
		if (i == part_num - 1)
		{
			now_size = last_size;
		}
		else
		{
			now_size = block_size;
		}
		std::cout << now_size << " " << n << std::endl;
		CHECK(hipMemcpy(d_a, mat1 + (long long)i * block_size * k, (long long)now_size * k * sizeof(float), hipMemcpyHostToDevice));
		CHECK(hipMemcpy(d_c, res + (long long)i * block_size * n, (long long)now_size * n * sizeof(float), hipMemcpyHostToDevice));
		CHECK(hipMemcpy(d_c_path, res_path + (long long)i * block_size * n, (long long)now_size * n * sizeof(int), hipMemcpyHostToDevice));

		const unsigned int GridDim_x = n / TILE_WIDTH + 1;
		const unsigned int GridDim_y = now_size / TILE_WIDTH + 1;

		hipLaunchKernelGGL(minplus_kernel_path, dim3(GridDim_x, GridDim_y), dim3(TILE_WIDTH, TILE_WIDTH), 0, 0, d_a, d_b, d_b_path, d_c, d_c_path, now_size, n, k);

		CHECK(hipMemcpy(res + (long long)i * block_size * n, d_c, (long long)now_size * n * sizeof(float), hipMemcpyDeviceToHost));
		CHECK(hipMemcpy(res_path + (long long)i * block_size * n, d_c_path, (long long)now_size * n * sizeof(int), hipMemcpyDeviceToHost));
	}
	hipFree(d_b);
	hipFree(d_b_path);
	hipFree(d_a);
	hipFree(d_c);
	hipFree(d_c_path);
}