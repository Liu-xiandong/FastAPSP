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

#ifndef UTIL_IMPROVED_H_
#define UTIL_IMPROVED_H_

#pragma once

#include <string.h>

#include <unordered_map>
#include <vector>
#include <algorithm>

#include "fap/utils/parameter.h"

namespace fap {

void MysubMatBuild_path(float *subMat, int *subMat_path,
    float *subGraph, int *subGraph_path, int *graph_id,
    int start, int sub_vertexs, int bdy_vertexs, int vertexs,
    int *st2ed, int *ed2st,
    int *adj_size, int *row_offset, int *col_val, float *weight) {
    // subMat build from csr Graph
    for (int i = 0; i < sub_vertexs; i++) {
        int ver = st2ed[i + start];
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int j = 0; j < adjcount; j++) {
            // the value of col is unmoved
            int neighbor = col_val[offset + j];
            if (graph_id[ver] != graph_id[neighbor])
                continue;
            float w = weight[offset + j];
            int index = ed2st[neighbor] - start;
            assert(index >= 0);
            assert(index < sub_vertexs);

            subMat[(int64_t)i * sub_vertexs + index] = w;
            subMat_path[(int64_t)i * sub_vertexs + index] = ver;
        }
    }

    // build from subGraph
    for (int i = 0; i < bdy_vertexs; i++) {
        for (int j = 0; j < sub_vertexs; j++) {
            int ver = st2ed[j + start];
            assert(ver >= 0);
            int64_t src = (int64_t)i * vertexs + ver;
            int64_t dst = (int64_t)i * sub_vertexs + j;
            subMat[dst] = subGraph[src];
            subMat_path[dst] = subGraph_path[src];
        }
    }

    // diag num
    for (int i = 0; i < sub_vertexs; i++) {
        int64_t dst = (int64_t)i * sub_vertexs + i;
        subMat[dst] = 0;
        subMat_path[dst] = st2ed[i + start];
    }
}

void MysubMatDecode_path(float *subMat, int *subMat_path,
    float *subGraph, int *subGraph_path,
    int start, int row, int col, int vertexs, int *st2ed) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int ver = st2ed[j + start];
            int64_t dst = (int64_t)i * vertexs + ver;
            subGraph[dst] = subMat[i * col + j];
            subGraph_path[dst] = subMat_path[i * col + j];
        }
    }
}

void MyMat1Build(float *mat1, float *subMat,
    int inner_num, int bdy_num, int sub_vertexs) {
    for (int i = 0; i < inner_num; i++) {
        int index = i + bdy_num;
        memcpy(mat1 + i * bdy_num, subMat + index * sub_vertexs,
            bdy_num * sizeof(float));
    }
}

}  // namespace fap

#endif
