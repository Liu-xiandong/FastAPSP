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

#ifndef UTIL_SSSP_H_
#define UTIL_SSSP_H_

#pragma once

#include <string.h>
#include <omp.h>

#include <functional>
#include <utility>
#include <algorithm>
#include <queue>
#include <vector>

#include "fap/utils/debug.h"
#include "fap/utils/parameter.h"

#ifdef WITH_CUDA
#include "fap/kernel/cuda/cuda_kernel.h"
#endif

#ifdef WITH_HIP
#include "fap/kernel/hip/hip_kernel.h"
#endif

using std::fill_n;
using std::vector;

namespace fap {

void dijkstra(int u, int vertexs, int *adj_size, int *row_offset,
    int *col_val, float *weight, float *dist) {
    fill_n(dist, vertexs, MAXVALUE);
    dist[u] = 0;

    std::vector<bool> st(vertexs, false);

    typedef std::pair<float, int> PDI;
    std::priority_queue<PDI, std::vector<PDI>, std::greater<PDI>> q;
    q.push({0, u});

    while (!q.empty()) {
        auto x = q.top();
        q.pop();
        int ver = x.second;
        float distance = x.first;

        if (st[ver])
            continue;
        st[ver] = true;

        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int i = 0; i < adjcount; i++) {
            int nextNode = col_val[offset + i];
            float w = weight[offset + i];
            if (dist[nextNode] > distance + w) {
                q.push({distance + w, nextNode});
                dist[nextNode] = distance + w;
            }
        }
    }
}

void dijkstra_path(int u, int vertexs, int *adj_size, int *row_offset,
    int *col_val, float *weight, float *dist, int *path) {
    fill_n(dist, vertexs, MAXVALUE);
    fill_n(path, vertexs, -1);
    dist[u] = 0;
    path[u] = u;

    std::vector<bool> st(vertexs, false);

    typedef std::pair<float, int> PDI;
    std::priority_queue<PDI, std::vector<PDI>, std::greater<PDI>> q;
    q.push({0, u});

    while (!q.empty()) {
        auto x = q.top();
        q.pop();
        int ver = x.second;
        float distance = x.first;

        if (st[ver])
            continue;
        st[ver] = true;

        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int i = 0; i < adjcount; i++) {
            int nextNode = col_val[offset + i];
            float w = weight[offset + i];
            if (dist[nextNode] > distance + w) {
                q.push({distance + w, nextNode});
                dist[nextNode] = distance + w;
                path[nextNode] = ver;
            }
        }
    }
}

struct Edge {
    int a, b;
    float c;
};

void johnson_spfa(int u, int vertexs, int edges, int *adj_size,
    int *row_offset, int *col_val, float *weight, float *dist) {
    vector<int> adj_size_reweight(vertexs + 1);
    vector<int> row_offset_reweight(vertexs + 2);
    vector<int> col_val_reweight(edges + vertexs);
    vector<float> weight_reweight(edges + vertexs);

    // add edges
    memmove((int *)&adj_size_reweight[0], adj_size, vertexs * sizeof(int));
    adj_size_reweight[vertexs] = vertexs;
    // row_offset
    memmove((int *)&row_offset_reweight[0], row_offset, (vertexs + 1) * sizeof(int));
    row_offset_reweight[vertexs + 1] = row_offset_reweight[vertexs] + vertexs;
    // col and weight
    memmove((int *)&col_val_reweight[0], col_val, edges * sizeof(int));
    for (int i = 0; i < vertexs; i++) {
        col_val_reweight[edges + i] = i;
    }
    memmove((float *)&weight_reweight[0], weight, edges * sizeof(float));
    for (int i = 0; i < vertexs; i++) {
        weight_reweight[edges + i] = 0;
    }

    vector<Edge> edge_tmp(edges + vertexs);
    int cnt = 0;
    for (int i = 0; i < vertexs + 1; i++) {
        int adjcount = adj_size_reweight[i];
        int offset = row_offset_reweight[i];
        for (int j = 0; j < adjcount; j++) {
            int nextNode = col_val_reweight[offset + j];
            float w = weight_reweight[offset + j];
            edge_tmp[cnt].a = i;
            edge_tmp[cnt].b = nextNode;
            edge_tmp[cnt].c = w;
            cnt++;
        }
    }

    // spfa
    fill_n(dist, vertexs + 1, MAXVALUE);
    dist[vertexs] = 0;

    vector<float> last(vertexs + 1);

    for (int i = 0; i < vertexs + 1; i++) {
        memcpy(last.data(), dist, (vertexs + 1) * sizeof(float));
        for (int j = 0; j < edges + vertexs; j++) {
            auto e = edge_tmp[j];
            if (dist[e.b] > last[e.a] + e.c) {
                dist[e.b] = last[e.a] + e.c;
            }
        }
    }

    return;
}

void johnson_reweight(const int vertexs, const int edges,
    int *adj_size, int *row_offset, int *col_val, float *weights) {
    vector<float> modify_dist(vertexs + 1);

    johnson_spfa(vertexs, vertexs, edges,
        adj_size, row_offset, col_val, weights, (float *)&modify_dist[0]);

    for (int i = 0; i < vertexs; i++) {
        int adjcount = adj_size[i];
        int offset = row_offset[i];
        for (int j = 0; j < adjcount; j++) {
            int nextNode = col_val[offset + j];
            weights[offset + j] =
                weights[offset + j] + modify_dist[i] - modify_dist[nextNode];
        }
    }
}

bool negtive_cycle(const int vertexs, const int edges,
    int *adj_size, int *row_offset, int *col_val, float *weights) {
    vector<float> dist(vertexs, 0);
    std::queue<int> q;
    std::vector<bool> st(vertexs, false);
    for (int i = 0; i < vertexs; i++) {
        q.push(i);
        st[i] = true;
    }
    vector<int> cnt(vertexs, 0);

    while (!q.empty()) {
        int t = q.front();
        q.pop();
        st[t] = false;

        int adjcount = adj_size[t];
        int offset = row_offset[t];
        for (int i = 0; i < adjcount; i++) {
            int nextNode = col_val[offset + i];
            float w = weights[offset + i];
            if (dist[nextNode] > dist[t] + w) {
                dist[nextNode] = dist[t] + w;
                cnt[nextNode] = cnt[t] + 1;
                if (cnt[nextNode] >= vertexs)
                    return true;
                if (!st[nextNode]) {
                    st[nextNode] = true;
                    q.push(nextNode);
                }
            }
        }
    }
    return false;
}

void batched_sssp_path(int *source_node, int source_node_num,
    int vertexs, int edges,
    int *adj_size, int *row_offset, int *col_val, float *weights,
    float *batched_dist, int *batched_path) {
#ifdef WITH_NVGRAPH
    batched_sssp_cuGraph(source_node, source_node_num, vertexs, edges,
                         adj_size, row_offset, col_val, weights,
                         batched_dist, batched_path);
#else
    #pragma omp parallel
    {
    #pragma omp for
        for (int i = 0; i < source_node_num; i++) {
            int ver = source_node[i];
            dijkstra_path(ver, vertexs, adj_size, row_offset, col_val, weights,
                batched_dist + (int64_t)i * vertexs,
                batched_path + (int64_t)i * vertexs);
        }
    }
#endif
}

void handle_boundry_path(float *subGraph, int *subGraph_path,
    int vertexs, int edges, int bdy_num,
    int *adj_size, int *row_offset, int *col_val, float *weights,
    int *st2ed, int offset) {
    // TODO(Liu_xiandong): The following macro judgment does not look intuitive
    // and needs to be changed later
#ifdef WITH_HIP
    handle_boundry_AMD_GPU(subGraph, vertexs, edges, bdy_num,
                            adj_size, row_offset, col_val, weight,
                            st2ed, offset);
#elif defined(WITH_CUDA) && !defined(WITH_NVGRAPH)
    handle_boundry_Nvidia_GPU(subGraph, vertexs, edges, bdy_num,
                            adj_size, row_offset, col_val, weights,
                            st2ed, offset);
#else
    int *source_node = new int[bdy_num];
    for (int i = 0; i < bdy_num; i++) {
        source_node[i] = st2ed[offset + i];
    }
    batched_sssp_path(source_node, bdy_num, vertexs, edges,
        adj_size, row_offset, col_val, weights, subGraph, subGraph_path);
    delete[] source_node;
#endif
}

}  // namespace fap

#endif
