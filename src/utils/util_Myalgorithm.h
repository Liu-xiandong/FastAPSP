#ifndef UTIL_IMPROVED_H_
#define UTIL_IMPROVED_H_

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <string.h>
#include "parameter.h"

void MysubMatBuild_path(float *subMat, int *subMat_path, float *subGraph, int *subGraph_path, int *graph_id,
                        int start, int sub_vertexs, int bdy_vertexs, int vertexs, int *st2ed, int *ed2st,
                        int *adj_size, int *row_offset, int *col_val, float *weight)
{
    //subMat build from csr Graph
    for (int i = 0; i < sub_vertexs; i++)
    {
        int ver = st2ed[i + start];
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int j = 0; j < adjcount; j++)
        {
            // the value of col is unmoved
            int neighbor = col_val[offset + j];
            if (graph_id[ver] != graph_id[neighbor])
                continue;
            float w = weight[offset + j];
            int index = ed2st[neighbor] - start;
            assert(index >= 0);
            assert(index < sub_vertexs);

            subMat[(long long)i * sub_vertexs + index] = w;
            subMat_path[(long long)i * sub_vertexs + index] = ver;


        }
    }

    // build from subGraph
    for (int i = 0; i < bdy_vertexs; i++)
    {
        for (int j = 0; j < sub_vertexs; j++)
        {
            int ver = st2ed[j + start];
            assert(ver >= 0);

            subMat[(long long)i * sub_vertexs + j] = subGraph[(long long)i * vertexs + ver];
            subMat_path[(long long)i * sub_vertexs + j] = subGraph_path[(long long)i * vertexs + ver];
        }
    }

    //diag num
    for (int i = 0; i < sub_vertexs; i++)
    {
        subMat[(long long)i * sub_vertexs + i] = 0;
        subMat_path[(long long)i * sub_vertexs + i] = st2ed[i + start];
    }
}

void MysubMatDecode_path(float *subMat, int *subMat_path, float *subGraph, int *subGraph_path,
                         int start, int row, int col, int vertexs, int *st2ed)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int ver = st2ed[j + start];
            subGraph[(long long)i * vertexs + ver] = subMat[i * col + j];
            subGraph_path[(long long)i * vertexs + ver] = subMat_path[i * col + j];
        }
    }
}

void MyMat1Build(float *mat1, float *subMat, int inner_num, int bdy_num, int sub_vertexs)
{
    for (int i = 0; i < inner_num; i++)
    {
        int index = i + bdy_num;
        memcpy(mat1 + i * bdy_num, subMat + index * sub_vertexs, bdy_num * sizeof(float));
    }
}

#endif