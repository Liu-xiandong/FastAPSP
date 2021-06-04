#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "parameter.h"

//CUDA RunTime API
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_functions.h>

#define TILE_WIDTH 32
#define MAXVALUE 1e8
#define PROFILER

__global__ void gemm_kernel_path(float *A, float *B, int *B_path, float *C, int *C_path, int m, int n, int k)
{
    //申请共享内存，存在于每个block中
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B_path[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Cvalue = MAXVALUE;
    int Cpath = -1;

    for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
    {
        if (row < m && t * TILE_WIDTH + tx < k)
            ds_A[tx][ty] = A[row * k + t * TILE_WIDTH + tx];
        else
            ds_A[tx][ty] = MAXVALUE;

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

    if ((row < m) && (col < n))
    {
        C[row * n + col] = Cvalue;
        C_path[row * n + col] = Cpath;
    }
}

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])
template <
    const int BLOCK_SIZE_M,         // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,         // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,         // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y,        // height of block of C that each thread calculate
    const int THREAD_SIZE_X,        // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    >
__global__ void MatrixMulCUDA6_path(
    float *__restrict__ A,
    float *__restrict__ B,
    int *__restrict__ B_path,
    float *__restrict__ C,
    int *__restrict__ C_path,
    const int K,
    const int N)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = ty * bszx + tx;

    // shared memory
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    __shared__ int Bs_path[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X];
    int accum_path[THREAD_SIZE_Y][THREAD_SIZE_X];

#pragma unroll
    for (int i = 0; i < THREAD_SIZE_Y; i++)
    {
#pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; j++)
        {
            accum[i][j] = MAXVALUE;
            accum_path[i][j] = -1;
        }
    }
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];
    int frag_b_path[THREAD_SIZE_X];

    // threads needed to load one row of tile
    // / 4 is because float4 is used
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K)
    {
// load A from global memory to shared memory
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
        {
            FETCH_FLOAT4(As[A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL + tile_idx,                    // col
                K)]);
        }

// load B from global memory to shared memory
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
        {
            FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                tile_idx + B_TILE_ROW_START + i, // row
                B_TILE_COL + BLOCK_SIZE_N * bx,  // col
                N)]);

            FETCH_INT4(Bs_path[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_INT4(B_path[OFFSET(
                tile_idx + B_TILE_ROW_START + i, // row
                B_TILE_COL + BLOCK_SIZE_N * bx,  // col
                N)]);
        }

        __syncthreads();

// compute c
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k)
        {
// load A from shared memory to register
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
            {
                frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y][k];
            }

// load B from shared memory to register
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
            {
                FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k][THREAD_SIZE_X * tx + thread_x]);
                FETCH_INT4(frag_b_path[thread_x]) = FETCH_INT4(Bs_path[k][THREAD_SIZE_X * tx + thread_x]);
            }

#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
            {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
                {
                    if (frag_a[thread_y] + frag_b[thread_x] < accum[thread_y][thread_x])
                    {
                        accum[thread_y][thread_x] = frag_a[thread_y] + frag_b[thread_x];
                        accum_path[thread_y][thread_x] = frag_b_path[thread_x];
                    }
                }
            }
        }
        __syncthreads();
    }

// store back to C
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
    {
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
        {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);

            FETCH_INT4(C_path[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_INT4(accum_path[thread_y][thread_x]);
        }
    }
}

__global__ void memset2D_path(float *mat, int dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    mat[row * dim + col] = MAXVALUE;
}

extern "C" void gemm_NVIDIA_path_backup(float *mat1, float *mat2, int *mat2_path,
                                        float *res, int *res_path, int m, int n, int k)
{
    float *d_a;
    float *d_b;
    float *d_c;
    int *d_b_path;
    int *d_c_path;
    checkCudaErrors(cudaMalloc(&d_a, (long long)m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b, (long long)k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c, (long long)m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b_path, (long long)k * n * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_c_path, (long long)m * n * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_a, mat1, (long long)m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, mat2, (long long)k * n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b_path, mat2_path, (long long)k * n * sizeof(int), cudaMemcpyHostToDevice));

    const unsigned int GridDim_x = n / TILE_WIDTH + 1;
    const unsigned int GridDim_y = m / TILE_WIDTH + 1;
    const unsigned int blockDim = TILE_WIDTH;

    gemm_kernel_path<<<dim3(GridDim_x, GridDim_y), dim3(blockDim, blockDim)>>>(d_a, d_b, d_b_path, d_c, d_c_path, m, n, k);

    checkCudaErrors(cudaMemcpy(res, d_c, (long long)m * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(res_path, d_c_path, (long long)m * n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_b_path);
    cudaFree(d_c_path);
}

extern "C" void gemm_NVIDIA_path(float *mat1, float *mat2, int *mat2_path,
                                 float *res, int *res_path, int m, int n, int k)
{
    float *d_a;
    float *d_b;
    float *d_c;
    int *d_b_path;
    int *d_c_path;

    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_X = 4;
    const int THREAD_SIZE_Y = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    long long m_padding = ((m % BLOCK_SIZE_M == 0) ? m : (m / BLOCK_SIZE_M + 1) * BLOCK_SIZE_M);
    long long n_padding = ((n % BLOCK_SIZE_N == 0) ? n : (n / BLOCK_SIZE_N + 1) * BLOCK_SIZE_N);
    long long k_padding = ((k % BLOCK_SIZE_K == 0) ? k : (k / BLOCK_SIZE_K + 1) * BLOCK_SIZE_K);

    checkCudaErrors(cudaMalloc(&d_a, m_padding * k_padding * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b, k_padding * n_padding * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c, m_padding * n_padding * sizeof(float)));

    checkCudaErrors(cudaMalloc(&d_b_path, k_padding * n_padding * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_c_path, m_padding * n_padding * sizeof(int)));

    memset2D_path<<<dim3(k_padding / 32, m_padding / 32), dim3(32, 32)>>>(d_a, k_padding);
    memset2D_path<<<dim3(n_padding / 32, k_padding / 32), dim3(32, 32)>>>(d_b, n_padding);
    memset2D_path<<<dim3(n_padding / 32, m_padding / 32), dim3(32, 32)>>>(d_c, n_padding);

    checkCudaErrors(cudaMemcpy2D(d_a, sizeof(float) * k_padding, mat1, sizeof(float) * k, sizeof(float) * k, m, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_b, sizeof(float) * n_padding, mat2, sizeof(float) * n, sizeof(float) * n, k, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_b_path, sizeof(int) * n_padding, mat2_path, sizeof(int) * n, sizeof(int) * n, k, cudaMemcpyHostToDevice));

    #ifdef PROFILER
    cudaEvent_t start, stop;   //declare
    cudaEventCreate(&start);   //set up
    cudaEventCreate(&stop);    //set up
    cudaEventRecord(start, 0); //start
    #endif

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 dimGrid(n_padding / BLOCK_SIZE_N, m_padding / BLOCK_SIZE_M);
    MatrixMulCUDA6_path<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
        <<<dimGrid, dimBlock>>>(d_a, d_b, d_b_path, d_c, d_c_path, k_padding, n_padding);

    #ifdef PROFILER
    cudaEventRecord(stop, 0); //finish
    cudaEventSynchronize(stop);
    float eTime;
    cudaEventElapsedTime(&eTime, start, stop);
    printf("the SECOND minplus flops is: %.2f Gflops\n", (float)m * n * k * 2.0 * 1.0e-9f / (eTime/1000.0f));
    #endif
    
    checkCudaErrors(cudaMemcpy2D(res, sizeof(float) * n, d_c, n_padding * sizeof(float), sizeof(float) * n, m, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(res_path, sizeof(int) * n, d_c_path, n_padding * sizeof(int), sizeof(int) * n, m, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_b_path);
    cudaFree(d_c_path);
}

__global__ void gemm_kernel(float *A, float *B, float *C, int m, int n, int k)
{
    //申请共享内存，存在于每个block中
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Cvalue = MAXVALUE;

    for (int t = 0; t < (k - 1) / TILE_WIDTH + 1; ++t)
    {
        if (row < m && t * TILE_WIDTH + tx < k)
            ds_A[tx][ty] = A[row * k + t * TILE_WIDTH + tx];
        else
            ds_A[tx][ty] = MAXVALUE;

        if (t * TILE_WIDTH + ty < k && col < n)
            ds_B[tx][ty] = B[(t * TILE_WIDTH + ty) * n + col];
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
    }

    if ((row < m) && (col < n))
    {
        C[row * n + col] = Cvalue;
    }
}

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
template <
    const int BLOCK_SIZE_M,         // height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,         // width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,         // width of block of C that each thread block calculate
    const int THREAD_SIZE_Y,        // height of block of C that each thread calculate
    const int THREAD_SIZE_X,        // width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    >
__global__ void MatrixMulCUDA6(
    float *__restrict__ A,
    float *__restrict__ B,
    float *__restrict__ C,
    const int K,
    const int N)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // size of thread block
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // thread id
    const int tid = ty * bszx + tx;

    // shared memory
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    // registers for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X];
    #pragma unroll
    for(int i = 0; i < THREAD_SIZE_Y; i++){
        #pragma unroll
        for(int j = 0 ; j < THREAD_SIZE_X; j++){
            accum[i][j] = MAXVALUE;
        }
    }
    // registers for A and B
    float frag_a[THREAD_SIZE_Y];
    float frag_b[THREAD_SIZE_X];

    // threads needed to load one row of tile
    // / 4 is because float4 is used
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // can not unroll since K can not be determined at this point
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K)
    {
        // load A from global memory to shared memory
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE)
        {
            FETCH_FLOAT4(As[A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
                BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                A_TILE_COL + tile_idx,                    // col
                K)]);
        }

        // load B from global memory to shared memory
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE)
        {
            FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                tile_idx + B_TILE_ROW_START + i, // row
                B_TILE_COL + BLOCK_SIZE_N * bx,  // col
                N)]);
        }

        __syncthreads();

        // compute c
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k)
        {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
            {
                frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y][k];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
            {
                FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k][THREAD_SIZE_X * tx + thread_x]);
            }

            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
            {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x)
                {
                    if(frag_a[thread_y] + frag_b[thread_x] < accum[thread_y][thread_x]){
                        accum[thread_y][thread_x] = frag_a[thread_y] + frag_b[thread_x];
                    }
                }
            }
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y)
    {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4)
        {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

__global__ void memset2D(float *mat, int dim){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    mat[row * dim + col] = MAXVALUE;
}

extern "C" void gemm_NVIDIA_GPU(float *mat1, float *mat2, float *res,
                                int m, int n, int k)
{
    /*
    float *d_a;
    float *d_b;
    float *d_c;
    checkCudaErrors(cudaMalloc(&d_a, m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b, k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c, m * n * sizeof(float)));

    // cudaEvent_t start, stop;
    // checkCudaErrors(cudaEventCreate(&start));
    // checkCudaErrors(cudaEventCreate(&stop));
    // float msecTotal = 0;
    // checkCudaErrors(cudaEventRecord(start));

    checkCudaErrors(cudaMemcpy(d_a, mat1, m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, mat2, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // const unsigned int GridDim_x = n / TILE_WIDTH + 1;
    // const unsigned int GridDim_y = m / TILE_WIDTH + 1;
    // const unsigned int blockDim = TILE_WIDTH;

    // //cudaLaunchKernelGGL(gemm_final_path_v2, dim3(GridDim_x, GridDim_y), dim3(blockDim, blockDim), 0, 0, d_a, d_b, d_c, d_path, m, n, k, d_ref);
    // gemm_kernel<<<dim3(GridDim_x, GridDim_y), dim3(blockDim, blockDim)>>>(d_a, d_b, d_c, m, n, k);

    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_X = 4;
    const int THREAD_SIZE_Y = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X + 1, BLOCK_SIZE_M / THREAD_SIZE_Y + 1);
    dim3 dimGrid(m / BLOCK_SIZE_N, n / BLOCK_SIZE_M);
    MatrixMulCUDA6<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
        <<<dimGrid, dimBlock>>>(d_a, d_b, d_c, k, m);

    // checkCudaErrors(cudaEventRecord(stop));
    // checkCudaErrors(cudaEventSynchronize(stop));
    // checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // float  gigaFlops =  ((float)2 * m * n * k * 1.0e-9f) / (msecTotal/ 1000.0f) ;
    // printf( "My min-plus Performance= %.2f GFlop/s, Time= %.3f msec.\n",gigaFlops,msecTotal);

    checkCudaErrors(cudaMemcpy(res, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    */

    float *d_a;
    float *d_b;
    float *d_c;
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_K = 32;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_X = 4;
    const int THREAD_SIZE_Y = 4;
    const bool ENABLE_DOUBLE_BUFFER = false;

    int m_padding = ((m%BLOCK_SIZE_M == 0) ? m: (m/BLOCK_SIZE_M + 1) * BLOCK_SIZE_M);
    int n_padding = ((n%BLOCK_SIZE_N == 0) ? n: (n/BLOCK_SIZE_N + 1) * BLOCK_SIZE_N);
    int k_padding = ((k%BLOCK_SIZE_K == 0) ? k: (k/BLOCK_SIZE_K + 1) * BLOCK_SIZE_K);

    checkCudaErrors(cudaMalloc(&d_a, m_padding * k_padding * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b, k_padding * n_padding * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c, m_padding * n_padding * sizeof(float)));

    memset2D<<< dim3(k_padding/32 , m_padding/32), dim3(32, 32) >>>(d_a, k_padding);
    memset2D<<< dim3(n_padding/32 , k_padding/32), dim3(32, 32) >>>(d_b, n_padding);
    memset2D<<< dim3(n_padding/32 , m_padding/32), dim3(32, 32) >>>(d_c, n_padding);

    // printf("the m n k padding is:( %d , %d , %d )\n",m_padding,n_padding,k_padding);
    // printf("the m n k is:( %d , %d , %d )\n", m, n, k);

    checkCudaErrors(cudaMemcpy2D(d_a, sizeof(float) * k_padding, mat1, sizeof(float) * k, sizeof(float) * k, m, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy2D(d_b, sizeof(float) * n_padding, mat2, sizeof(float) * n, sizeof(float) * n, k, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    checkCudaErrors(cudaEventRecord(start));

    dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X , BLOCK_SIZE_M / THREAD_SIZE_Y );
    dim3 dimGrid( n_padding / BLOCK_SIZE_N, m_padding / BLOCK_SIZE_M);
    MatrixMulCUDA6<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
        <<<dimGrid, dimBlock>>>(d_a, d_b, d_c, k_padding, n_padding);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    float  gigaFlops =  ((float)2 * m * n * k * 1.0e-9f) / (msecTotal/ 1000.0f) ;
    printf( "My FIRST min-plus Performance= %.2f GFlop/s, Time= %.3f msec.\n",gigaFlops,msecTotal);

    checkCudaErrors(cudaMemcpy2D(res, sizeof(float) * n, d_c, n_padding*sizeof(float), sizeof(float) * n, m, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}