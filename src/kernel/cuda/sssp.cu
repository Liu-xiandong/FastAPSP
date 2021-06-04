#include <bits/stdc++.h>
#include <float.h>
#include <iomanip>
#include <limits.h>
#include "nvgraph.h"

//CUDA RunTime API
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_functions.h>

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
    //float **sssp_h;
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

extern "C" void batched_sssp_cuGraph(int *source_node, int source_node_num, int vertexs, int edges,
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