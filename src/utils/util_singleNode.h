#ifndef UTIL_SINGLENODE_H_
#define UTIL_SINGLENODE_H_

#include <vector>
#include <unordered_map>
#include "MatMul.h"
#include "debug.h"
#include "parameter.h"
#include <time.h>
#include <sys/time.h>

using std::fill_n;
using std::unordered_map;

#define TIMER

void singNode_subMatBuild_path(float *subMat, int *subMat_path, int *adj_size, int *row_offset, int *col_val, float *weight,
                               int *st2ed, int *ed2st, int sub_vertexs, int sub_start, int *C_BlockVer_offset, int MyId)
{
    for (int i = 0; i < sub_vertexs; i++)
    {
        int ver = st2ed[i + sub_start];
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int j = 0; j < adjcount; j++)
        {
            int neighbor = ed2st[col_val[offset + j]];
            int left = C_BlockVer_offset[MyId];
            int right = C_BlockVer_offset[MyId] + sub_vertexs;
            if (neighbor < left || neighbor >= right)
                continue;
            float w = weight[offset + j];
            subMat[(long long)i * sub_vertexs + neighbor - left] = w;
            subMat_path[(long long)i * sub_vertexs + neighbor - left] = ver;
        }
    }

    for (int i = 0; i < sub_vertexs; i++)
    {
        subMat[i * sub_vertexs + i] = 0;
        subMat_path[i * sub_vertexs + i] = st2ed[i + sub_start];
    }
}

void singNode_graph2bdyMat_Build(int K, int *C_BlockBdy_num, int *C_BlockBdy_offset,
                                 unordered_map<int, int> &graph2bdyMat)
{
    int cnt = 0;
    for (int i = 1; i <= K; i++)
    {
        int st = C_BlockBdy_offset[i];
        int len = C_BlockBdy_num[i];
        for (int index = st; index <= st + len - 1; index++)
        {
            graph2bdyMat[index] = cnt++;
        }
    }
}

void singNode_bdyMatBuild_path(float *bdyMat, int *bdyMat_path, float *subMat, int *subMat_path, int *adj_size, int *row_offset, int *col_val, float *weight,
                               int K, int *C_BlockBdy_num, int *C_BlockBdy_offset, int MyId,
                               int *st2ed, int *ed2st, int sub_start, int bdy_vertexs, int sub_vertexs,
                               unordered_map<int, int> &graph2bdyMat)
{
    //added from csr-Graph
    int subBdy_vertexs = C_BlockBdy_num[MyId];
    for (int i = 0; i < subBdy_vertexs; i++)
    {
        int ver = st2ed[i + sub_start];
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int j = 0; j < adjcount; j++)
        {
            int neighbor = ed2st[col_val[offset + j]];
            if (graph2bdyMat.find(neighbor) == graph2bdyMat.end())
                continue;
            float w = weight[offset + j];
            bdyMat[(long long)i * bdy_vertexs + graph2bdyMat[neighbor]] = w;
            bdyMat_path[(long long)i * bdy_vertexs + graph2bdyMat[neighbor]] = ver;
        }
    }

    //added from subMat
    for (int i = 0; i < subBdy_vertexs; i++)
    {
        for (int j = 0; j < subBdy_vertexs; j++)
        {
            int st = C_BlockBdy_offset[MyId];
            int index = st + j;
            bdyMat[(long long)i * bdy_vertexs + graph2bdyMat[index]] = subMat[i * sub_vertexs + j];
            bdyMat_path[(long long)i * bdy_vertexs + graph2bdyMat[index]] = subMat_path[i * sub_vertexs + j];
        }
    }

    //init the diag element
}

void singNode_bdyMat2subMat_path(float *bdyMat, int *bdyMat_path, float *subMat, int *subMat_path,
                                 int K, int *C_BlockBdy_num, int *C_BlockBdy_offset, int MyId,
                                 int bdy_vertexs, int sub_vertexs,
                                 unordered_map<int, int> &graph2bdyMat)
{
    int subBdy_vertexs = C_BlockBdy_num[MyId];
    for (int i = 0; i < subBdy_vertexs; i++)
    {
        for (int j = 0; j < subBdy_vertexs; j++)
        {
            int st = C_BlockBdy_offset[MyId];
            int index = st + j;
            subMat[i * sub_vertexs + j] = bdyMat[(long long)i * bdy_vertexs + graph2bdyMat[index]];
            subMat_path[i * sub_vertexs + j] = bdyMat_path[(long long)i * bdy_vertexs + graph2bdyMat[index]];
        }
    }
}

void singNode_handle_diffgraph_path(float *subGraph, int *subGraph_path, int idx, int idy, int n,
                                    int *C_BlockVer_num, int *C_BlockVer_offset,
                                    int *C_BlockBdy_num, int *C_BlockBdy_offset,
                                    float *bdyMat, float *subMat_x, float *subMat_y,
                                    int *subMat_y_path, int bdy_vertexs, int K,
                                    unordered_map<int, int> &graph2bdyMat)
{
#ifdef TIMER
    struct timeval begin_total, end_total;
    struct timeval begin_computation1, end_computation1;
    struct timeval begin_computation2, end_computation2;
    struct timeval begin_data_move, end_data_move;
    struct timeval begin_fill_n, end_fill_n;

    double elapsedTime_total = 0;
    double elapsedTime_computation1 = 0;
    double elapsedTime_computation2 = 0;
    double elapsedTime_data_move = 0;
    double elapsedTime_fill_n = 0;
#endif

#ifdef TIMER
    gettimeofday(&begin_total, NULL);
#endif
    int idx_num = C_BlockVer_num[idx];
    int idx_bdy_num = C_BlockBdy_num[idx];
    int idy_bdy_num = C_BlockBdy_num[idy];
    int idy_num = C_BlockVer_num[idy];

    float *mat1 = (float *)malloc(idx_num * idx_bdy_num * sizeof(float));
    float *mat2 = (float *)malloc(idx_bdy_num * idy_bdy_num * sizeof(float));
    float *mat3 = (float *)malloc(idy_bdy_num * idy_num * sizeof(float));
    int *mat3_path = (int *)malloc(idy_bdy_num * idy_num * sizeof(int));
    float *res = (float *)malloc(idx_num * idy_num * sizeof(float));
    int *res_path = (int *)malloc(idx_num * idy_num * sizeof(int));
    float *tmp = (float *)malloc(idx_num * idy_bdy_num * sizeof(float));
    int *tmp_path = (int *)malloc(idx_num * idy_bdy_num * sizeof(int));

#ifdef TIMER
    gettimeofday(&begin_fill_n, NULL);
#endif
    fill_n(mat1, idx_num * idx_bdy_num, MAXVALUE);
    fill_n(mat2, idx_bdy_num * idy_bdy_num, MAXVALUE);
    fill_n(mat3, idy_bdy_num * idy_num, MAXVALUE);
    fill_n(tmp, idx_num * idy_bdy_num, MAXVALUE);
    //fill_n(res, idx_num * idy_num, MAXVALUE);
    fill_n(tmp_path, idx_num * idy_bdy_num, -1);
    //fill_n(res_path, idx_num * idy_num, -1);
    fill_n(mat3_path, idy_bdy_num * idy_num, -1);
#ifdef TIMER
    gettimeofday(&end_fill_n, NULL);
    elapsedTime_fill_n += (end_fill_n.tv_sec - begin_fill_n.tv_sec) + (end_fill_n.tv_usec - begin_fill_n.tv_usec) / 1000000.0;
#endif

    //fill mat1
    for (int i = 0; i < idx_num; i++)
    {
        for (int j = 0; j < idx_bdy_num; j++)
        {
            mat1[i * idx_bdy_num + j] = subMat_x[i * idx_num + j];
        }
    }

    //fill mat2
    int idy_index = C_BlockBdy_offset[idy];
    for (int i = 0; i < idx_bdy_num; i++)
    {
        for (int j = 0; j < idy_bdy_num; j++)
        {
            mat2[i * idy_bdy_num + j] = bdyMat[(long long)i * bdy_vertexs + graph2bdyMat[idy_index + j]];
        }
    }

    //fill mat3
    for (int i = 0; i < idy_bdy_num; i++)
    {
        for (int j = 0; j < idy_num; j++)
        {
            mat3[i * idy_num + j] = subMat_y[i * idy_num + j];
            mat3_path[i * idy_num + j] = subMat_y_path[i * idy_num + j];
        }
    }

#ifdef TIMER
    gettimeofday(&begin_computation1, NULL);
#endif
    min_plus_path(mat1, mat2, tmp, tmp_path, idx_num, idy_bdy_num, idx_bdy_num);
#ifdef TIMER
    gettimeofday(&end_computation1, NULL);
    elapsedTime_computation1 += (end_computation1.tv_sec - begin_computation1.tv_sec) + (end_computation1.tv_usec - begin_computation1.tv_usec) / 1000000.0;
#endif

#ifdef TIMER
    gettimeofday(&begin_computation2, NULL);
#endif
    min_plus_path_advanced(tmp, mat3, mat3_path, res, res_path, idx_num, idy_num, idy_bdy_num);
#ifdef TIMER
    gettimeofday(&end_computation2, NULL);
    elapsedTime_computation2 += (end_computation2.tv_sec - begin_computation2.tv_sec) + (end_computation2.tv_usec - begin_computation2.tv_usec) / 1000000.0;
#endif

#ifdef TIMER
    gettimeofday(&begin_data_move, NULL);
#endif
    //res fill back to arc
    for (int i = 0; i < idx_num; i++)
    {
        for (int j = 0; j < idy_num; j++)
        {
            subGraph[(long long)i * n + idy_index + j] = res[i * idy_num + j];
            subGraph_path[(long long)i * n + idy_index + j] = res_path[i * idy_num + j];
        }
    }
#ifdef TIMER
    gettimeofday(&end_data_move, NULL);
    elapsedTime_data_move += (end_data_move.tv_sec - begin_data_move.tv_sec) + (end_data_move.tv_usec - begin_data_move.tv_usec) / 1000000.0;
#endif

#ifdef TIMER
    gettimeofday(&end_total, NULL);
    elapsedTime_total += (end_total.tv_sec - begin_total.tv_sec) + (end_total.tv_usec - begin_total.tv_usec) / 1000000.0;
#endif

#ifdef TIMER
    printf("(total:%lf) (compu1:%lf) (compu2:%lf) (data_move:%lf) (fill_n:%lf)\n ", elapsedTime_total, elapsedTime_computation1, elapsedTime_computation2, elapsedTime_data_move,elapsedTime_fill_n);
#endif

    free(mat1);
    free(mat2);
    free(mat3);
    free(res);
    free(tmp);
    free(res_path);
    free(tmp_path);
}

#endif