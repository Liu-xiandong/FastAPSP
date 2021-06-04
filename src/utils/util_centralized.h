#ifndef UTIL_CENTRALIZED_H_
#define UTIL_CENTRALIZED_H_

#include <unordered_map>
#include <vector>
#include <algorithm>
#include "floyd.h"
#include "MatMul.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sssp.h"
#include "parameter.h"
#include "util_preprocess.h"

using std::cout;
using std::fill_n;
using std::string;
using std::vector;

// balanced
void balanced_tasks(int *C_BlockBdy_num, std::unordered_map<int, std::vector<int>> &process_tasks,
                    int *tasks_array, int *tasks_num, int *tasks_offset, int K, int P)
{

    // every tasks is allocated to process
    for (int i = 0; i < K; i++)
    {
        tasks_array[i] = i + 1;
    }
    std::random_shuffle(tasks_array, tasks_array + K);
    debug_array(tasks_array, K);
    int num = K / P;
    int last = K - num * (P - 1);
    assert(last > 0);

    for (int i = 0; i < P; i++)
    {
        if (i == P - 1)
            tasks_num[i] = last;
        else
            tasks_num[i] = num;
    }
    for (int i = 0; i < P; i++)
    {
        tasks_offset[i] = i * num;
    }
}

//-----------------step1----------------------
// fill the subGraph_path with Graph represented by csr format
void subGraphBuild_path(float *subGraph, int *subGraph_path, int vertexs,
                        int *adj_size, int *row_offset, int *col_val, float *weight,
                        int *st2ed, int *ed2st, int sub_vertexs, int sub_start)
{
    for (int i = 0; i < sub_vertexs; i++)
    {
        int ver = st2ed[i + sub_start];
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int j = 0; j < adjcount; j++)
        {
            int neighbor = ed2st[col_val[offset + j]];
            float w = weight[offset + j];
            subGraph[(long long)i * vertexs + neighbor] = w;
            subGraph_path[(long long)i * vertexs + neighbor] = ver;
        }
    }
}

// reorder the subGraph_path from index 1 to index n
void subGraphReorder_path(float *subGraph, int *subGraph_path, int sub_vertexs, int vertexs, int *st2ed)
{
    for (int i = 0; i < sub_vertexs; i++)
    {
        float *t = new float[vertexs];
        int *t_path = new int[vertexs];
        for (int j = 0; j < vertexs; j++)
        {
            int index = st2ed[j];
            t[index] = subGraph[(long long)i * vertexs + j];
            t_path[index] = subGraph_path[(long long)i * vertexs + j];
        }
        memcpy(subGraph + (long long)i * vertexs, t, vertexs * sizeof(float));
        memcpy(subGraph_path + (long long)i * vertexs, t_path, vertexs * sizeof(int));
        delete[] t;
        delete[] t_path;
    }
}

// fill the subMat used in Floyd-Warshall from the subGraph_path
void subMatBuild_path(float *subMat, int *subMat_path, float *subGraph, int *subGraph_path,
                      int start, int sub_vertexs, int vertexs, int *st2ed)
{
    for (int i = 0; i < sub_vertexs; i++)
    {
        for (int j = 0; j < sub_vertexs; j++)
        {
            subMat[i * sub_vertexs + j] = subGraph[(long long)i * vertexs + j + start];
            subMat_path[i * sub_vertexs + j] = subGraph_path[(long long)i * vertexs + j + start];
        }
    }

    for (int i = 0; i < sub_vertexs; i++)
    {
        subMat[i * sub_vertexs + i] = 0;
        subMat_path[i * sub_vertexs + i] = st2ed[i + start];
    }
}

void subMatDecode_path(float *subMat, int *subMat_path, float *subGraph, int *subGraph_path,
                       int start, int sub_vertexs, int vertexs)
{
    for (int i = 0; i < sub_vertexs; i++)
    {
        for (int j = 0; j < sub_vertexs; j++)
        {
            subGraph[(long long)i * vertexs + j + start] = subMat[i * sub_vertexs + j];
            subGraph_path[(long long)i * vertexs + j + start] = subMat_path[i * sub_vertexs + j];
        }
    }
}

//-----------------step2----------------------
// build the bdyMat in each subGraph
void bdyMatBuild(float *bdyMat, float *subGraph,
                 int K, int subBdy_vertexs, int vertexs,
                 int *C_BlockBdy_num, int *C_BlockBdy_offset)
{
    int cnt = 0;
    for (int i = 0; i < subBdy_vertexs; i++)
    {
        for (int j = 1; j <= K; j++)
        {
            int st = C_BlockBdy_offset[j];
            int len = C_BlockBdy_num[j];
            for (int index = st; index <= st + len - 1; index++)
            {
                bdyMat[cnt++] = subGraph[(long long)i * vertexs + index];
            }
        }
    }
}

void bdyMatBuild_path(float *bdyMat, float *subGraph,
                      int *bdyMat_path, int *subGraph_path,
                      int K, int subBdy_vertexs, int vertexs,
                      int *C_BlockBdy_num, int *C_BlockBdy_offset)
{
    int cnt = 0;
    for (int i = 0; i < subBdy_vertexs; i++)
    {
        for (int j = 1; j <= K; j++)
        {
            int st = C_BlockBdy_offset[j];
            int len = C_BlockBdy_num[j];
            for (int index = st; index <= st + len - 1; index++)
            {
                bdyMat[cnt] = subGraph[(long long)i * vertexs + index];
                bdyMat_path[cnt] = subGraph_path[(long long)i * vertexs + index];
                cnt++;
            }
        }
    }
}

void bdyMessageBuild(int *C_bdyMat_num, int *C_bdyMat_offset,
                     int K, int *C_BlockBdy_num, int &bdy_vertexs)
{
    bdy_vertexs = 0;
    for (int i = 1; i <= K; i++)
        bdy_vertexs += C_BlockBdy_num[i];
    int cnt = 0;
    for (int i = 1; i <= K; i++)
    {
        C_bdyMat_offset[i] = cnt;
        cnt += bdy_vertexs * C_BlockBdy_num[i];
        C_bdyMat_num[i] = bdy_vertexs * C_BlockBdy_num[i];
    }
}

void bdyMatDecode_path(float *bdyMat, int *bdyMat_path,
                       float *subGraph, int *subGraph_path,
                       int K, int subBdy_vertexs, int vertexs,
                       int *C_BlockBdy_num, int *C_BlockBdy_offset)
{
    int cnt = 0;
    for (int i = 0; i < subBdy_vertexs; i++)
    {
        for (int j = 1; j <= K; j++)
        {
            int st = C_BlockBdy_offset[j];
            int len = C_BlockBdy_num[j];
            for (int index = st; index <= st + len - 1; index++)
            {
                subGraph[(long long)i * vertexs + index] = bdyMat[cnt];
                subGraph_path[(long long)i * vertexs + index] = bdyMat_path[cnt];
                cnt++;
            }
        }
    }
}

//-----------------step3----------------------
void subMatMessageBuild(int *C_subMat_num, int *C_subMat_offset,
                        int K, int *C_BlockVer_num, long long &ALL_subMat_num, const int padding)
{
    int cnt = 0;
    for (int i = 1; i <= K; i++)
    {
        C_subMat_offset[i] = cnt;
        long long len = (long long)C_BlockVer_num[i] * C_BlockVer_num[i];
        long long len_padding = (len % padding == 0) ? len : (len / padding + 1) * padding;
        cnt += static_cast<int>(len_padding / padding);
        C_subMat_num[i] = static_cast<int>(len_padding / padding);
    }
    ALL_subMat_num = cnt;
}

void subMatMessageBuild(int K, int vertexs,
                        int *C_subGraph_offset, int *C_BlockVer_num)
{
    int cnt = 0;
    for (int i = 1; i <= K; i++)
    {
        C_subGraph_offset[i] = cnt;
        cnt += C_BlockVer_num[i] * vertexs;
    }
}

void handle_diffgraph_path(int idx, int idy, int n,
                           int *C_BlockVer_num, int *C_BlockVer_offset,
                           int *C_BlockBdy_num, int *C_BlockBdy_offset,
                           float *subMat, float *subGraph, int *subGraph_path,
                           float *All_subMat, int *All_subMat_path,
                           int *C_subMat_offset, const int padding)
{
    int idx_num = C_BlockVer_num[idx];
    int idx_bdy_num = C_BlockBdy_num[idx];
    int idy_bdy_num = C_BlockBdy_num[idy];
    int idy_num = C_BlockVer_num[idy];

    float *mat1 = (float *)malloc(idx_num * idx_bdy_num * sizeof(float));
    float *mat2 = (float *)malloc(idx_bdy_num * idy_bdy_num * sizeof(float));
    float *mat3 = (float *)malloc(idy_bdy_num * idy_num * sizeof(float));
    int *mat3_path = (int *)malloc(idy_bdy_num * idy_num * sizeof(int));
    float *res = (float *)malloc(idx_num * idy_num * sizeof(float));
    float *tmp = (float *)malloc(idx_num * idy_bdy_num * sizeof(float));
    int *res_path = (int *)malloc(idx_num * idy_num * sizeof(int));
    int *tmp_path = (int *)malloc(idx_num * idy_bdy_num * sizeof(int));

    fill_n(mat1, idx_num * idx_bdy_num, MAXVALUE);
    fill_n(mat2, idx_bdy_num * idy_bdy_num, MAXVALUE);
    fill_n(mat3, idy_bdy_num * idy_num, MAXVALUE);
    fill_n(tmp, idx_num * idy_bdy_num, MAXVALUE);
    fill_n(res, idx_num * idy_num, MAXVALUE);
    fill_n(mat3_path, idy_bdy_num * idy_num, -1);
    fill_n(tmp_path, idx_num * idy_bdy_num, -1);
    fill_n(res_path, idx_num * idy_num, -1);

    //fill mat1
    for (int i = 0; i < idx_num; i++)
    {
        for (int j = 0; j < idx_bdy_num; j++)
        {
            mat1[i * idx_bdy_num + j] = subMat[i * idx_num + j];
        }
    }

    //fill mat2
    int idy_index = C_BlockBdy_offset[idy];
    for (int i = 0; i < idx_bdy_num; i++)
    {
        for (int j = 0; j < idy_bdy_num; j++)
        {
            mat2[i * idy_bdy_num + j] = subGraph[(long long)i * n + idy_index + j];
        }
    }

    //fill mat3
    long long All_subMat_index = (long long)C_subMat_offset[idy] * padding;
    for (int i = 0; i < idy_bdy_num; i++)
    {
        for (int j = 0; j < idy_num; j++)
        {
            mat3[i * idy_num + j] = All_subMat[All_subMat_index + i * idy_num + j];
            mat3_path[i * idy_num + j] = All_subMat_path[All_subMat_index + i * idy_num + j];
        }
    }

    min_plus_path(mat1, mat2, tmp, tmp_path, idx_num, idy_bdy_num, idx_bdy_num);
    min_plus_path_advanced(tmp, mat3, mat3_path, res, res_path, idx_num, idy_num, idy_bdy_num);

    //res fill back to arc
    for (int i = 0; i < idx_num; i++)
    {
        for (int j = 0; j < idy_num; j++)
        {
            subGraph[(long long)i * n + idy_index + j] = res[i * idy_num + j];
            subGraph_path[(long long)i * n + idy_index + j] = res_path[i * idy_num + j];
        }
    }

    free(mat1);
    free(mat2);
    free(mat3);
    free(res);
    free(tmp);
    free(res_path);
    free(tmp_path);
}

void local2global_path(int *path, int row, int col, int *st2ed, int offset)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int tmp = path[i * col + j];
            if (tmp == -1)
                continue;
            path[i * col + j] = st2ed[tmp + offset];
        }
    }
}

#endif