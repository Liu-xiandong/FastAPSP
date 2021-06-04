#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

//CUDA RunTime API
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_functions.h>

#define TILE_WIDTH 32
#define MAX_VALUE 1e8

__device__ __forceinline__ float atomicMinFloat0(float *addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMax((int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void initKernel(float *d_weightRow, float weight, float *d_weightRowTemp, float weightTemp, bool *d_mask, int k, int num_node)
{

    int tid = threadIdx.x;
    int x = blockIdx.x * blockDim.x + tid;
    if (x < num_node)
    {
        d_weightRow[x] = weight;
        d_weightRowTemp[x] = weightTemp;
        d_mask[x] = false;
        if (x == k)
        {
            d_weightRow[k] = 0;
            d_weightRowTemp[k] = 0;
            d_mask[k] = true;
        }
    }
}

__global__ void dijkstraKernel1(int *d_rowOffsetArc, int *d_colValueArc, float *d_weightArc, float *d_weightRow, float *d_weightRowTemp, bool *d_mask, int num_node)
{
    int tid = threadIdx.x;
    int x = blockIdx.x * blockDim.x + tid;
    if (x < num_node && d_mask[x] == true)
    {
        d_mask[x] = false;
        float dis = d_weightRow[x];
        int rowOffset = d_rowOffsetArc[x + 1];
        for (int j = d_rowOffsetArc[x]; j < rowOffset; j++)
        {
            atomicMinFloat0(&d_weightRowTemp[d_colValueArc[j]], dis + d_weightArc[j]);
        }
    }
}
__global__ void dijkstraKernel2(float *d_weightRow, float *d_weightRowTemp, bool *d_mask, int num_node)
{
    int tid = threadIdx.x;
    int x = blockIdx.x * blockDim.x + tid;
    if (x < num_node)
    {
        if (d_weightRow[x] > d_weightRowTemp[x])
        {
            d_weightRow[x] = d_weightRowTemp[x];
            d_mask[x] = true;
        }
        d_weightRowTemp[x] = d_weightRow[x];
    }
}

extern "C" void multi_source_Nvidia_sssp(int numFrom, int *fromNode, int vertexs, int edges, int *rowOffsetArc, int *colValueArc, float *weightArc, float *shortLenTable)
{
    int numberOfBlock;
    //在GPU中分配空间
    float *d_weightRow;
    float *d_weightRowTemp;

    int *d_rowOffsetArc;
    int *d_colValueArc;
    float *d_weightArc;
    bool *d_mask;
    bool *mask;
    mask = (bool *)malloc(vertexs * sizeof(bool));

    float *d_res;

    checkCudaErrors(cudaMalloc((void **)&d_rowOffsetArc, vertexs * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_colValueArc, edges * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_weightArc, edges * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_weightRow, vertexs * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_weightRowTemp, vertexs * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_mask, vertexs * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void **)&d_res, numFrom * vertexs * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_rowOffsetArc, rowOffsetArc, vertexs * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_colValueArc, colValueArc, edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_weightArc, weightArc, edges * sizeof(float), cudaMemcpyHostToDevice));

    const int blockDim = 32;
    numberOfBlock = ceil((float)vertexs / (float)blockDim);
    bool isEmpty;
    for (int k = 0; k < numFrom; k++)
    {
        isEmpty = false;
        //cudaLaunchKernelGGL(initKernel, dim3(numberOfBlock, 1), dim3(blockDim, 1), 0, 0, d_res + k * vertexs, MAX_VALUE, d_weightRowTemp, MAX_VALUE, d_mask, fromNode[k], vertexs);
        initKernel<<<dim3(numberOfBlock, 1), dim3(blockDim, 1)>>>(d_res + k * vertexs, MAX_VALUE, d_weightRowTemp, MAX_VALUE, d_mask, fromNode[k], vertexs);
        while (!isEmpty)
        {
            //cudaLaunchKernelGGL(dijkstraKernel1, dim3(numberOfBlock, 1), dim3(blockDim, 1), 0, 0, d_rowOffsetArc, d_colValueArc, d_weightArc, d_res + k * vertexs, d_weightRowTemp, d_mask, vertexs);
            //cudaLaunchKernelGGL(dijkstraKernel2, dim3(numberOfBlock, 1), dim3(blockDim, 1), 0, 0, d_res + k * vertexs, d_weightRowTemp, d_mask, vertexs);
            dijkstraKernel1<<<dim3(numberOfBlock, 1), dim3(blockDim, 1)>>>(d_rowOffsetArc, d_colValueArc, d_weightArc, d_res + k * vertexs, d_weightRowTemp, d_mask, vertexs);
            dijkstraKernel2<<<dim3(numberOfBlock, 1), dim3(blockDim, 1)>>>(d_res + k * vertexs, d_weightRowTemp, d_mask, vertexs);
            checkCudaErrors(cudaMemcpy(mask, d_mask, vertexs * sizeof(bool), cudaMemcpyDeviceToHost));
            isEmpty = true;
            for (int i = 0; i < vertexs; i++)
            {
                //	printf("mask[%d]=%d\n",i,mask[i]);
                if (mask[i] != false)
                {
                    isEmpty = false;
                }
            }
        }
        //printf("fromNode[%d] = %d\n",k,fromNode[k]);
        //checkCudaErrors(cudaMemcpy(&shortLenTable[k * vertexs], d_weightRow, vertexs * sizeof(float), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaMemcpy(shortLenTable, d_res, numFrom * vertexs * sizeof(float), cudaMemcpyDeviceToHost));

    //将数据free
    checkCudaErrors(cudaFree(d_rowOffsetArc));
    checkCudaErrors(cudaFree(d_colValueArc));
    checkCudaErrors(cudaFree(d_weightArc));
    checkCudaErrors(cudaFree(d_weightRow));
    checkCudaErrors(cudaFree(d_weightRowTemp));
    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_res));
    free(mask);
}

extern "C" void handle_boundry_Nvidia_GPU(float *subGraph, int vertexs, int edges, int bdy_num,
                                       int *adj_size, int *row_offset, int *col_val, float *weight,
                                       int *st2ed, int offset)
{
    int *sources = new int[bdy_num];
    for (int i = 0; i < bdy_num; i++)
    {
        int ver = st2ed[offset + i];
        sources[i] = ver;
    }

    multi_source_Nvidia_sssp(bdy_num, sources, vertexs, edges, row_offset, col_val, weight, subGraph);
    delete[] sources;
}