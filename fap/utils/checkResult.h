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

#ifndef LAB_GEO_APSP_MASTER_FAP_UTILS_CHECKRESULT_H_
#define LAB_GEO_APSP_MASTER_FAP_UTILS_CHECKRESULT_H_

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <unordered_map>
#include <vector>
#include <algorithm>

#include "fap/kernel/batch_sssp.h"
#include "fap/utils/parameter.h"

using std::vector;

namespace fap {

// check the dist and path is right
// the dist and path is the part of the matrix
// source stands for the vertexs need to check
bool check_ans(float *dist, int *path, int *source, int source_num, int vertexs,
               int *adj_size, int *row_offset, int *col_val,
               float *weight, int *graph_id) {
    time_t t;
    srand((unsigned)time(&t));
    unsigned int local_seed = time(NULL);
    int check_cases = 20;
    while (check_cases--) {
        // the source vertexs needed to be test
        int ver_index = rand_r(&local_seed) % source_num;
        // the real vertex id
        int ver = source[ver_index];

        int check_col_cases = 10;
        while (check_col_cases--) {
            // the real vertexs id
            int des = rand_r(&local_seed) % vertexs;

            float now_dist = dist[ver_index * vertexs + des];
            int bridge = path[ver_index * vertexs + des];

            vector<float> check_dist(vertexs);
            dijkstra(ver, vertexs, adj_size,
                row_offset, col_val, weight, check_dist.data());
            float ver2bridge = check_dist[bridge];

            // unreached vertexs
            if (now_dist > MAXVALUE / 10 && check_dist[des] > MAXVALUE / 10)
                continue;

            const float eps = 1e-5;
            if (fabs(now_dist - check_dist[des]) > eps) {
                printf("the dist is wrong!\n");
                printf("%f - %f \n", check_dist[des], now_dist);
                if (graph_id[ver] == graph_id[des]) {
                    printf("wrong in the floyd\n");
                } else {
                    printf("maybe wrong in the minplus\n");
                }
                return false;
            }
        }
    }
    return true;
}

}  // namespace fap

#endif  // LAB_GEO_APSP_MASTER_FAP_UTILS_CHECKRESULT_H_
