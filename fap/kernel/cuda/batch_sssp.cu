#include <bits/stdc++.h>
#include <float.h>
#include <iomanip>
#include <limits.h>
#include <iostream>

//CUDA RunTime API
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#ifdef WITH_NVGRAPH
#include "nvgraph.h"
#endif

#define TILE_WIDTH 32
#define MAX_VALUE 1e8

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

// ############## use the nvGraph API ##############

#ifdef WITH_NVGRAPH
void check(nvgraphStatus_t status)
{
    if (status != NVGRAPH_STATUS_SUCCESS)
    {
        printf("ERROR : %d\n", status);
        exit(0);
    }
}

template <class I, class T>
void csr_tocsc(const I n_row,
               const I n_col,
               const I Ap[],
               const I Aj[],
               const T Ax[],
               I Bp[],
               I Bi[],
               T Bx[])
{
    const I nnz = Ap[n_row];

    //compute number of non-zero entries per column of A
    std::fill(Bp, Bp + n_col, 0);

    for (I n = 0; n < nnz; n++)
    {
        Bp[Aj[n]]++;
    }

    //cumsum the nnz per column to get Bp[]
    for (I col = 0, cumsum = 0; col < n_col; col++)
    {
        I temp = Bp[col];
        Bp[col] = cumsum;
        cumsum += temp;
    }
    Bp[n_col] = nnz;

    for (I row = 0; row < n_row; row++)
    {
        for (I jj = Ap[row]; jj < Ap[row + 1]; jj++)
        {
            I col = Aj[jj];
            I dest = Bp[col];

            Bi[dest] = row;
            Bx[dest] = Ax[jj];

            Bp[col]++;
        }
    }

    for (I col = 0, last = 0; col <= n_col; col++)
    {
        I temp = Bp[col];
        Bp[col] = last;
        last = temp;
    }
}

void mul_sssp_nvGraph(int *source, int sourceN, int n, int nnz, float *weights_h, int *destination_offsets_h, int *source_indices_h,float* dist)
{
    const size_t vertex_numsets = 1;
    const size_t edge_numsets = 1;
    void **vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t *vertex_dimT;

    // Init host data
    float *sssp_1_h = (float *)malloc(n * sizeof(float));
    vertex_dim = (void **)malloc(vertex_numsets * sizeof(void *));
    vertex_dimT = (cudaDataType_t *)malloc(vertex_numsets * sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0] = (void *)sssp_1_h;
    vertex_dimT[0] = CUDA_R_32F;

    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr(handle, &graph));
    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void *)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void *)weights_h, 0));
    // Solve
    for (int i = 0; i < sourceN; i++)
    {
        //int source_vert = source;
        check(nvgraphSssp(handle, graph, 0, &source[i], 0));
        // Get and print result
        check(nvgraphGetVertexData(handle, graph, (void *)&dist[i*n], 0));
        //debug_array((float *)&sssp_h_tmp[0], n);
    }

    //Clean
    free(sssp_1_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
}

void batched_sssp_cuGraph(int *source_node, int source_node_num, int vertexs, int edges,
                            int *adj_size, int *row_offset, int *col_val, float *weights, 
                            float *batched_dist, int *batched_path)
{
    //CSC
    float *weightCSC = (float *)malloc(edges * sizeof(float));
    int *rowValueCSC = (int *)malloc(edges * sizeof(int));
    int *colOffsetCSC = (int *)malloc((vertexs + 1) * sizeof(int));

    csr_tocsc(vertexs, vertexs, row_offset, col_val, weights, colOffsetCSC, rowValueCSC, weightCSC);

    mul_sssp_nvGraph(source_node, source_node_num, vertexs, edges, weightCSC, colOffsetCSC, rowValueCSC, batched_dist);

    free(weightCSC);
    free(rowValueCSC);
    free(colOffsetCSC);
}
#endif

// ############## use the manual kernel ##############

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

void multi_source_Nvidia_sssp(int numFrom, int *fromNode, int vertexs, int edges, int *rowOffsetArc, int *colValueArc, float *weightArc, float *shortLenTable)
{
    int numberOfBlock;
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
        initKernel<<<dim3(numberOfBlock, 1), dim3(blockDim, 1)>>>(d_res + k * vertexs, MAX_VALUE, d_weightRowTemp, MAX_VALUE, d_mask, fromNode[k], vertexs);
        while (!isEmpty)
        {
            dijkstraKernel1<<<dim3(numberOfBlock, 1), dim3(blockDim, 1)>>>(d_rowOffsetArc, d_colValueArc, d_weightArc, d_res + k * vertexs, d_weightRowTemp, d_mask, vertexs);
            dijkstraKernel2<<<dim3(numberOfBlock, 1), dim3(blockDim, 1)>>>(d_res + k * vertexs, d_weightRowTemp, d_mask, vertexs);
            checkCudaErrors(cudaMemcpy(mask, d_mask, vertexs * sizeof(bool), cudaMemcpyDeviceToHost));
            isEmpty = true;
            for (int i = 0; i < vertexs; i++)
            {
                if (mask[i] != false)
                {
                    isEmpty = false;
                }
            }
        }
    }
    checkCudaErrors(cudaMemcpy(shortLenTable, d_res, numFrom * vertexs * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_rowOffsetArc));
    checkCudaErrors(cudaFree(d_colValueArc));
    checkCudaErrors(cudaFree(d_weightArc));
    checkCudaErrors(cudaFree(d_weightRow));
    checkCudaErrors(cudaFree(d_weightRowTemp));
    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_res));
    free(mask);
}

void handle_boundry_Nvidia_GPU(float *subGraph, int vertexs, int edges, int bdy_num,
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
