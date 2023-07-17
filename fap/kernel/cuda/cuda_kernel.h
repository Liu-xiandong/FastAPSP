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

// sssp_kernel
void batched_sssp_cuGraph(
    int *source_node, int source_node_num, int vertexs, int edges,
    int *adj_size, int *row_offset, int *col_val, float *weights,
    float *batched_dist, int *batched_path);

void handle_boundry_Nvidia_GPU(
    float *subGraph, int vertexs, int edges, int bdy_num,
    int *adj_size, int *row_offset, int *col_val, float *weight,
    int *st2ed, int offset);

// floyd_kernel
void floyd_NVIDIA_GPU(int num_node, float *arc);

void floyd_GPU_Nvidia_path(int num_node, float *arc, int *path);

void floyd_path_A_Nvidia(float *A, int *A_path,
    const int row, const int col, float *diag, int *diag_path);
void floyd_path_B_Nvidia(float *B, int *B_path,
    const int row, const int col, float *diag, int *diag_path);
void floyd_min_plus_Nvidia(float *mat1, float *mat2,
    int *mat2_path, float *res, int *res_path, int m, int n, int k);

// minplus_kernel
void minplus_NVIDIA_GPU(float *mat1, float *mat2, float *res,
    int M, int N, int K);
void minplus_NVIDIA_path(float *mat1, float *mat2, int *mat2_path,
    float *res, int *res_path, int m, int n, int k);
