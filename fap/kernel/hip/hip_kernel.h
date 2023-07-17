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

void handle_boundry_AMD_GPU(
    float *subGraph, int vertexs, int edges, int bdy_num,
    int *adj_size, int *row_offset, int *col_val, float *weight,
    int *st2ed, int offset);

// floyd_kernel
void floyd_AMD_GPU(int num_node, float *arc);
void floyd_AMD_path(int num_node, float *arc, int *path);

void floyd_path_A_AMD(float *A, int *A_path,
    const int row, const int col, float *diag, int *diag_path);
void floyd_path_B_AMD(float *B, int *B_path,
    const int row, const int col, float *diag, int *diag_path);
void floyd_min_plus_AMD(float *mat1, float *mat2,
    int *mat2_path, float *res, int *res_path, int m, int n, int k);

void floyd_minplus_partition_AMD(float *mat1, float *mat2,
    int *mat2_path, float *res, int *res_path,
    int m, int n, int k, const int part_num);

// minplus_kernel
void minplus_AMD_GPU(float *mat1, float *mat2, float *res,
    int M, int N, int K);
void minplus_AMD_path(float *mat1, float *mat2, int *mat2_path,
    float *res, int *res_path, int m, int n, int k);
void minplus_AMD_partition(float *mat1, float *mat2, int *mat2_path,
    int m, int n, int k, const int part_num,
    float *subMat, int *subMat_path,
    int start, int sub_vertexs, int vertexs, int *st2ed);
