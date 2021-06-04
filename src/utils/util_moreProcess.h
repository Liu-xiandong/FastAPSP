#ifndef UTIL_MOREPROCESS_H_
#define UTIL_MOREPROCESS_H_

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

using std::cout;
using std::fill_n;
using std::string;
using std::vector;

void build_bdymat_message(unordered_map<int, unordered_map<int, int>> &bdy_mat_tile_size,
                          unordered_map<int, unordered_map<int, vector<int>>> &bdy_mat_vertexs,
                          unordered_map<int, unordered_map<int, int>> &bdy_mat_offset,
                          int *st2ed, int *C_BlockBdy_num, int *C_BlockBdy_offset,
                          int dist_col, int K, int vertexs)
{
    for (int i = 1; i <= K; i++)
    {
        const int bdy_num = C_BlockBdy_num[i];
        int tile_size = floor((double)bdy_num / dist_col);
        int last_size = bdy_num - tile_size * (dist_col - 1);
        assert(last_size > 0);
        int offset = C_BlockBdy_offset[i];
        int cnt = 0;
        int bdy_offset = 0;
        for (int j = 0; j < dist_col; j++)
        {
            // source number of sssp task
            int now_bdy_size;
            if (j == dist_col - 1)
            {
                now_bdy_size = last_size;
            }
            else
            {
                now_bdy_size = tile_size;
            }
            bdy_mat_tile_size[i][j] = now_bdy_size;
            // source number array of sssp task
            cnt += j * tile_size;
            vector<int> bdy_vertexs;
            for (int index = offset + cnt; index < offset + cnt + now_bdy_size; index++)
            {
                bdy_vertexs.push_back(st2ed[index]);
            }
            bdy_mat_vertexs[i][j] = bdy_vertexs;
            // offset in bdy_mat
            bdy_offset = j * tile_size * vertexs;
            bdy_mat_offset[i][j] = bdy_offset;
        }
    }
}

void build_sssp_comm_message(int *bdy_mat_num, int *bdy_mat_suboffset,
                             unordered_map<int, unordered_map<int, int>> &bdy_mat_tile_size,
                             const int dist_col, const int dist_x, const int vertexs)
{
    int cnt = 0;
    for (int i = 0; i < dist_col; i++)
    {
        bdy_mat_suboffset[i] = cnt;
        bdy_mat_num[i] = bdy_mat_tile_size[dist_x][i] * vertexs;
        cnt += bdy_mat_num[i];
    }
}

void build_subMat_message(unordered_map<int, unordered_map<int, int>> &floyd_tile_size,
                          unordered_map<int, unordered_map<int, int>> &subMat_st2ed_offset,
                          const int K, int *C_BlockVer_num, int *C_BlockVer_offset,
                          const int dist_col, const int floyd_block_size)
{
    for (int i = 1; i <= K; i++)
    {
        const int sub_vertexs = C_BlockVer_num[i];
        int tile_size_tmp = (int)floor((double)sub_vertexs / dist_col);
        // tile_size padding
        int tile_size = (tile_size_tmp % floyd_block_size == 0) ? tile_size_tmp : (tile_size_tmp / floyd_block_size) * floyd_block_size;
        assert(tile_size < sub_vertexs);
        int last_size = sub_vertexs - tile_size * (dist_col - 1);
        assert(last_size > 0);

        int floyd_offset = C_BlockVer_offset[i];
        for (int j = 0; j < dist_col; j++)
        {
            // distributed floyd size
            int now_floyd_size;
            if (j == dist_col - 1)
            {
                now_floyd_size = last_size;
            }
            else
            {
                now_floyd_size = tile_size;
            }
            floyd_tile_size[i][j] = now_floyd_size;
            //  floyd offset in st2ed array
            subMat_st2ed_offset[i][j] = floyd_offset;
            floyd_offset += tile_size;
        }
    }
}

void build_distributed_subMat(float *subMat, int *subMat_path, const int now_floyd_size,
                              int vertexs, int sub_vertexs,
                              int *st2ed, int *ed2st, int *graph_id, const int now_floyd_offset,
                              int *adj_size, int *row_offset, int *col_val, float *weight,
                              const int dist_y, const int bdy_num, const int now_floyd_tile_size,
                              const int start, float *bdy_mat, int *bdy_mat_path, const int dist_x)
{
    //subMat build from csr Graph
    for (int i = 0; i < now_floyd_size; i++)
    {
        int ver = st2ed[start + now_floyd_offset + i];
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int j = 0; j < adjcount; j++)
        {
            // the value of col is unmoved
            int neighbor = col_val[offset + j];
            if (graph_id[ver] != graph_id[neighbor])
            {
                continue;
            }
            float w = weight[offset + j];
            int index = ed2st[neighbor] - start;
            assert(graph_id[ver] == dist_x);
            assert(index >= 0);
            assert(index < sub_vertexs);
            subMat[(long long)i * sub_vertexs + index] = w;
            subMat_path[(long long)i * sub_vertexs + index] = now_floyd_offset + i;
        }
    }

    //build from bdy_mat
    int last_process = (int)ceil((double)bdy_num / now_floyd_tile_size);
    if (dist_y < last_process)
    {
        int now_bdy_size;
        if (dist_y == last_process - 1)
        {
            now_bdy_size = bdy_num - dist_y * now_floyd_tile_size;
        }
        else
        {
            now_bdy_size = now_floyd_tile_size;
        }
        long long bdy_mat_offset = dist_y * now_floyd_tile_size * vertexs;
        for (int i = 0; i < now_bdy_size; i++)
        {
            for (int j = 0; j < sub_vertexs; j++)
            {
                int ver = st2ed[j + start];
                subMat[(long long)i * sub_vertexs + j] = bdy_mat[bdy_mat_offset + (long long)i * vertexs + ver];
                int tmp = ed2st[bdy_mat_path[bdy_mat_offset + (long long)i * vertexs + ver]] - start;
                subMat_path[(long long)i * sub_vertexs + j] = tmp;
            }
        }
    }

    //diag num
    for (int i = 0; i < now_floyd_size; i++)
    {
        subMat[(long long)i * sub_vertexs + now_floyd_offset + i] = 0;
        subMat_path[(long long)i * sub_vertexs + now_floyd_offset + i] = now_floyd_offset + i;
    }
}

//-----------------step3----------------------
// build related message in Floyd-Warshall process
void build_floyd_diag(float *floyd_diag, int *floyd_diag_path, const int floyd_block_size,
                      float *subMat, int *subMat_path, const int row, const int col, const int diag_x, const int diag_y)
{
    for (int i = 0; i < floyd_block_size; i++)
    {
        for (int j = 0; j < floyd_block_size; j++)
        {
            if (i + diag_x >= row || j + diag_y >= col)
            {
                floyd_diag[i * floyd_block_size + j] = MAXVALUE;
                floyd_diag_path[i * floyd_block_size + j] = -1;
                continue;
            }
            floyd_diag[i * floyd_block_size + j] = subMat[(i + diag_x) * col + (j + diag_y)];
            floyd_diag_path[i * floyd_block_size + j] = subMat_path[(i + diag_x) * col + (j + diag_y)];
        }
    }
}

void build_floyd_B(float *floyd_B, int *floyd_B_path, int B_row, int B_col,
                   float *subMat, int *subMat_path, int row, int col, int offset)
{
    for (int i = 0; i < B_row; i++)
    {
        for (int j = 0; j < B_col; j++)
        {
            if (i + offset >= row || j >= col)
            {
                floyd_B[i * B_col + j] = MAXVALUE;
                floyd_B_path[i * B_col + j] = -1;
                continue;
            }
            floyd_B[i * B_col + j] = subMat[(i + offset) * col + j];
            floyd_B_path[i * B_col + j] = subMat_path[(i + offset) * col + j];
        }
    }
}

void build_floyd_A(float *floyd_A, int *floyd_A_path, int A_row, int A_col,
                   float *subMat, int *subMat_path, int row, int col, int offset)
{
    for (int i = 0; i < A_row; i++)
    {
        for (int j = 0; j < A_col; j++)
        {
            if (i >= row || offset + j >= col)
            {
                floyd_A[i * A_col + j] = MAXVALUE;
                floyd_A_path[i * A_col + j] = -1;
                continue;
            }
            floyd_A[i * A_col + j] = subMat[i * col + offset + j];
            floyd_A_path[i * A_col + j] = subMat_path[i * col + offset + j];
        }
    }
}

void build_minplus_A(float *minplus_A, int *minplus_A_path, int A_row, int A_col,
                     float *subMat, int *subMat_path, int row, int col, int offset)
{
    for (int i = 0; i < A_row; i++)
    {
        for (int j = 0; j < A_col; j++)
        {
            if (i + offset >= row || j >= col)
            {
                minplus_A[i * A_col + j] = MAXVALUE;
                minplus_A_path[i * A_col + j] = -1;
                continue;
            }
            minplus_A[i * A_col + j] = subMat[(i + offset) * col + j];
            minplus_A_path[i * A_col + j] = subMat_path[(i + offset) * col + j];
        }
    }
}

#endif
