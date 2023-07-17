#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "util_Myalgorithm.h"
#include "parameter.h"

//hip RunTime API
#include "hip/hip_runtime.h"
#include <hc_defines.h>

#define TILE_WIDTH 32
//#define MAXVALUE 1e8

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

__device__ uint64_t inline readtime()
{
    uint64_t clock;
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                 "s_memtime %0\n\t"
                 "s_waitcnt lgkmcnt(0)\n\t"
                 : "=s"(clock));
    return clock;
}

__global__ void minplus_kernel(float *A, float *B, int *B_path, float *C, int *C_path, unsigned int m, unsigned int n, unsigned int k)
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

    for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
    {
        if (row < m && (t * TILE_WIDTH + tx < k))
            ds_A[tx][ty] = A[row * k + t * TILE_WIDTH + tx];
        else
            ds_A[tx][ty] = MAXVALUE;

        if ((t * TILE_WIDTH + ty < k) && col < n)
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

    if ((row < m) && (col < n))
    {
        C[row * n + col] = Cvalue;
        C_path[row * n + col] = Cpath;
    }
}

void minplus_AMD_path(float *mat1, float *mat2, int *mat2_path,
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

    CHECK(hipMalloc(&d_a, m * k * sizeof(float)));
    CHECK(hipMalloc(&d_b, k * n * sizeof(float)));
    CHECK(hipMalloc(&d_c, m * n * sizeof(float)));
    CHECK(hipMalloc(&d_b_path, k * n * sizeof(int)));
    CHECK(hipMalloc(&d_c_path, m * n * sizeof(int)));

    CHECK(hipMemcpy(d_a, mat1, m * k * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, mat2, k * n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b_path, mat2_path, k * n * sizeof(int), hipMemcpyHostToDevice));

    const unsigned int GridDim_x = n / TILE_WIDTH + 1;
    const unsigned int GridDim_y = m / TILE_WIDTH + 1;
    const unsigned int blockDim = TILE_WIDTH;

    hipLaunchKernelGGL(minplus_kernel, dim3(GridDim_x, GridDim_y), dim3(blockDim, blockDim), 0, 0, d_a, d_b, d_b_path, d_c, d_c_path, m, n, k);

    CHECK(hipMemcpy(res, d_c, m * n * sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(res_path, d_c_path, m * n * sizeof(int), hipMemcpyDeviceToHost));

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    hipFree(d_b_path);
    hipFree(d_c_path);
}

void minplus_AMD_partition(float *mat1, float *mat2, int *mat2_path,
                                   int m, int n, int k, const int part_num,
                                   float *subMat, int *subMat_path,
                                   int start, int sub_vertexs, int vertexs, int *st2ed)
{
    assert(part_num > 1);
    int block_size = m / part_num;
    int last_size = m - block_size * (part_num);

    float *d_b;
    int *d_b_path;
    CHECK(hipMalloc(&d_b, (long long)k * n * sizeof(float)));
    CHECK(hipMalloc(&d_b_path, (long long)k * n * sizeof(int)));
    CHECK(hipMemcpy(d_b, mat2, (long long)k * n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b_path, mat2_path, (long long)k * n * sizeof(int), hipMemcpyHostToDevice));

    int max_size = max(block_size, last_size);
    float *res = new float[max_size * n];
    int *res_path = new int[max_size * n];

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

        float *d_a;
        float *d_c;
        int *d_c_path;
        CHECK(hipMalloc(&d_a, (long long)now_size * k * sizeof(float)));
        CHECK(hipMalloc(&d_c, (long long)now_size * n * sizeof(float)));
        CHECK(hipMalloc(&d_c_path, (long long)now_size * n * sizeof(int)));

        CHECK(hipMemcpy(d_a, mat1 + (long long)i * block_size * k, (long long)now_size * k * sizeof(float), hipMemcpyHostToDevice));

        const long long GridDim_x = n / TILE_WIDTH + 1;
        const long long GridDim_y = now_size / TILE_WIDTH + 1;
        const long long blockDim = TILE_WIDTH;

        hipLaunchKernelGGL(minplus_kernel, dim3(GridDim_x, GridDim_y), dim3(blockDim, blockDim), 0, 0, d_a, d_b, d_b_path, d_c, d_c_path, now_size, n, k);

        CHECK(hipMemcpy(res + (long long)i * block_size * n, d_c, (long long)now_size * n * sizeof(float), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(res_path + (long long)i * block_size * n, d_c_path, (long long)now_size * n * sizeof(int), hipMemcpyHostToDevice));

        hipFree(d_a);
        hipFree(d_c);
        hipFree(d_c_path);

        int offset = i * block_size * sub_vertexs;
        MysubMatDecode_path(subMat + offset, subMat_path + offset,
                            res, res_path, start, now_size, sub_vertexs, vertexs, st2ed);
    }
    hipFree(d_b);
    hipFree(d_b_path);
    delete[] res;
    delete[] res_path;
}

__global__ void minplus_kernel(float *A, float *B, float *C, int m, int n, int k)
{
	//申请共享内存，存在于每个block中
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	float Cvalue = MAXVALUE;

	for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
	{
		if (row < m && t * TILE_WIDTH + tx < k)
			ds_A[tx][ty] = A[(long long)row * k + t * TILE_WIDTH + tx];
		else
			ds_A[tx][ty] = MAXVALUE;

		if (t * TILE_WIDTH + ty < k && col < n)
			ds_B[tx][ty] = B[((long long)t * TILE_WIDTH + ty) * n + col];
		else
			ds_B[tx][ty] = MAXVALUE;

		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i)
		{
			if (Cvalue > ds_A[i][ty] + ds_B[tx][i])
			{
				Cvalue = ds_A[i][ty] + ds_B[tx][i];
			}
		}
		__syncthreads();

		if ((row < m) && (col < n))
		{
			C[(long long)row * n + col] = Cvalue;
		}
	}
}

void minplus_AMD_GPU(float *mat1, float *mat2, float *res,
							 int m, int n, int k)
{
	float *d_a;
	float *d_b;
	float *d_c;
	CHECK(hipMalloc(&d_a, (unsigned int)m * k * sizeof(float)));
	CHECK(hipMalloc(&d_b, (unsigned int)k * n * sizeof(float)));
	CHECK(hipMalloc(&d_c, (unsigned int)m * n * sizeof(float)));

	CHECK(hipMemcpy(d_a, mat1, (unsigned int)m * k * sizeof(float), hipMemcpyHostToDevice));
	CHECK(hipMemcpy(d_b, mat2, (unsigned int)k * n * sizeof(float), hipMemcpyHostToDevice));

	const unsigned int GridDim_x = n / TILE_WIDTH + 1;
	const unsigned int GridDim_y = m / TILE_WIDTH + 1;
	const unsigned int blockDim = TILE_WIDTH;

	hipLaunchKernelGGL(minplus_kernel, dim3(GridDim_x, GridDim_y), dim3(blockDim, blockDim), 0, 0, d_a, d_b, d_c, m, n, k);
	//minplus_kernel<<<dim3(GridDim_x, GridDim_y), dim3(blockDim, blockDim)>>>(d_a, d_b, d_c, m, n, k);

	CHECK(hipMemcpy(res, d_c, (unsigned int)m * n * sizeof(float), hipMemcpyDeviceToHost));

	hipFree(d_a);
	hipFree(d_b);
	hipFree(d_c);
}