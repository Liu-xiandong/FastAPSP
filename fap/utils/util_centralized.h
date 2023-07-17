// Copyright 2023 The Fap Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UTIL_CENTRALIZED_H_
#define UTIL_CENTRALIZED_H_

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <string>

#include "fap/kernel/floyd.h"
#include "fap/kernel/minplus.h"
#include "fap/kernel/batch_sssp.h"
#include "fap/utils/parameter.h"
#include "fap/utils/util_preprocess.h"

using std::cout;
using std::fill_n;
using std::string;
using std::vector;

namespace fap {

// -----------------step1----------------------
// fill the subGraph_path with Graph represented by csr format
void subGraphBuild_path(float *subGraph, int *subGraph_path, int vertexs,
                int *adj_size, int *row_offset, int *col_val, float *weight,
                int *st2ed, int *ed2st, int sub_vertexs, int sub_start) {
    for (int i = 0; i < sub_vertexs; i++) {
        int ver = st2ed[i + sub_start];
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int j = 0; j < adjcount; j++) {
            int neighbor = ed2st[col_val[offset + j]];
            float w = weight[offset + j];
            subGraph[(int64_t)i * vertexs + neighbor] = w;
            subGraph_path[(int64_t)i * vertexs + neighbor] = ver;
        }
    }
}

// reorder the subGraph_path from index 1 to index n
void subGraphReorder_path(float *subGraph, int *subGraph_path,
        int sub_vertexs, int vertexs, int *st2ed) {
    for (int i = 0; i < sub_vertexs; i++) {
        float *t = new float[vertexs];
        int *t_path = new int[vertexs];
        for (int j = 0; j < vertexs; j++) {
            int index = st2ed[j];
            t[index] = subGraph[(int64_t)i * vertexs + j];
            t_path[index] = subGraph_path[(int64_t)i * vertexs + j];
        }
        memcpy(subGraph + (int64_t)i * vertexs, t, vertexs * sizeof(float));
        memcpy(subGraph_path + (int64_t)i * vertexs,
            t_path, vertexs * sizeof(int));
        delete[] t;
        delete[] t_path;
    }
}

// fill the subMat used in Floyd-Warshall from the subGraph_path
void subMatBuild_path(float *subMat, int *subMat_path,
        float *subGraph, int *subGraph_path,
        int start, int sub_vertexs, int vertexs, int *st2ed) {
    for (int i = 0; i < sub_vertexs; i++) {
        for (int j = 0; j < sub_vertexs; j++) {
            int64_t src = (int64_t)i * vertexs + j + start;
            int64_t dst = i * sub_vertexs + j;
            subMat[src] = subGraph[dst];
            subMat_path[src] = subGraph_path[dst];
        }
    }

    for (int i = 0; i < sub_vertexs; i++) {
        subMat[i * sub_vertexs + i] = 0;
        subMat_path[i * sub_vertexs + i] = st2ed[i + start];
    }
}

void subMatDecode_path(float *subMat, int *subMat_path,
        float *subGraph, int *subGraph_path,
        int start, int sub_vertexs, int vertexs) {
    for (int i = 0; i < sub_vertexs; i++) {
        for (int j = 0; j < sub_vertexs; j++) {
            int64_t src = i * sub_vertexs + j;
            int64_t dst = (int64_t)i * vertexs + j + start;
            subGraph[dst] = subMat[src];
            subGraph_path[dst] = subMat_path[src];
        }
    }
}

// -----------------step2----------------------
// build the bdyMat in each subGraph
void bdyMatBuild(float *bdyMat, float *subGraph,
                 int K, int subBdy_vertexs, int vertexs,
                 int *C_BlockBdy_num, int *C_BlockBdy_offset) {
    int cnt = 0;
    for (int i = 0; i < subBdy_vertexs; i++) {
        for (int j = 1; j <= K; j++) {
            int st = C_BlockBdy_offset[j];
            int len = C_BlockBdy_num[j];
            for (int index = st; index <= st + len - 1; index++) {
                bdyMat[cnt++] = subGraph[(int64_t)i * vertexs + index];
            }
        }
    }
}

void bdyMatBuild_path(float *bdyMat, float *subGraph,
                      int *bdyMat_path, int *subGraph_path,
                      int K, int subBdy_vertexs, int vertexs,
                      int *C_BlockBdy_num, int *C_BlockBdy_offset) {
    int cnt = 0;
    for (int i = 0; i < subBdy_vertexs; i++) {
        for (int j = 1; j <= K; j++) {
            int st = C_BlockBdy_offset[j];
            int len = C_BlockBdy_num[j];
            for (int index = st; index <= st + len - 1; index++) {
                int64_t src = (int64_t)i * vertexs + index;
                bdyMat[cnt] = subGraph[src];
                bdyMat_path[cnt] = subGraph_path[src];
                cnt++;
            }
        }
    }
}

void bdyMessageBuild(int *C_bdyMat_num, int *C_bdyMat_offset,
            int K, int *C_BlockBdy_num, int &bdy_vertexs) {
    bdy_vertexs = 0;
    for (int i = 1; i <= K; i++)
        bdy_vertexs += C_BlockBdy_num[i];
    int cnt = 0;
    for (int i = 1; i <= K; i++) {
        C_bdyMat_offset[i] = cnt;
        cnt += bdy_vertexs * C_BlockBdy_num[i];
        C_bdyMat_num[i] = bdy_vertexs * C_BlockBdy_num[i];
    }
}

void bdyMatDecode_path(float *bdyMat, int *bdyMat_path,
                       float *subGraph, int *subGraph_path,
                       int K, int subBdy_vertexs, int vertexs,
                       int *C_BlockBdy_num, int *C_BlockBdy_offset) {
    int cnt = 0;
    for (int i = 0; i < subBdy_vertexs; i++) {
        for (int j = 1; j <= K; j++) {
            int st = C_BlockBdy_offset[j];
            int len = C_BlockBdy_num[j];
            for (int index = st; index <= st + len - 1; index++) {
                subGraph[(int64_t)i * vertexs + index] = bdyMat[cnt];
                subGraph_path[(int64_t)i * vertexs + index] = bdyMat_path[cnt];
                cnt++;
            }
        }
    }
}

// -----------------step3----------------------
void subMatMessageBuild(int *C_subMat_num, int *C_subMat_offset,
    int K, int *C_BlockVer_num, int64_t &ALL_subMat_num, const int padding) {
    int cnt = 0;
    for (int i = 1; i <= K; i++) {
        C_subMat_offset[i] = cnt;
        int64_t len = (int64_t)C_BlockVer_num[i] * C_BlockVer_num[i];
        int64_t len_padding =
            (len % padding == 0) ? len : (len / padding + 1) * padding;
        cnt += static_cast<int>(len_padding / padding);
        C_subMat_num[i] = static_cast<int>(len_padding / padding);
    }
    ALL_subMat_num = cnt;
}

void subMatMessageBuild(int K, int vertexs,
                        int *C_subGraph_offset, int *C_BlockVer_num) {
    int cnt = 0;
    for (int i = 1; i <= K; i++) {
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

    // fill mat1
    for (int i = 0; i < idx_num; i++) {
        for (int j = 0; j < idx_bdy_num; j++) {
            mat1[i * idx_bdy_num + j] = subMat[i * idx_num + j];
        }
    }

    // fill mat2
    int idy_index = C_BlockBdy_offset[idy];
    for (int i = 0; i < idx_bdy_num; i++) {
        for (int j = 0; j < idy_bdy_num; j++) {
            int64_t src = (int64_t)i * n + idy_index + j;
            int64_t dst = i * idy_bdy_num + j;
            mat2[dst] = subGraph[src];
        }
    }

    // fill mat3
    int64_t All_subMat_index = (int64_t)C_subMat_offset[idy] * padding;
    for (int i = 0; i < idy_bdy_num; i++) {
        for (int j = 0; j < idy_num; j++) {
            int64_t src = All_subMat_index + i * idy_num + j;
            mat3[i * idy_num + j] = All_subMat[src];
            mat3_path[i * idy_num + j] = All_subMat_path[src];
        }
    }

    min_plus_path(mat1, mat2, tmp, tmp_path, idx_num, idy_bdy_num, idx_bdy_num);
    min_plus_path_advanced(tmp, mat3, mat3_path, res, res_path,
        idx_num, idy_num, idy_bdy_num);

    // res fill back to arc
    for (int i = 0; i < idx_num; i++) {
        for (int j = 0; j < idy_num; j++) {
            int64_t src = i * idy_num + j;
            int64_t dst = (int64_t)i * n + idy_index + j;
            subGraph[dst] = res[src];
            subGraph_path[dst] = res_path[src];
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

void local2global_path(int *path, int row, int col, int *st2ed, int offset) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int tmp = path[i * col + j];
            if (tmp == -1)
                continue;
            path[i * col + j] = st2ed[tmp + offset];
        }
    }
}

}  // namespace fap

#endif
