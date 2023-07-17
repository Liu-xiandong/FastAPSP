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

#ifndef FLOYD_H_
#define FLOYD_H_

#pragma once

#include <algorithm>
#include <iostream>

#include "fap/utils/parameter.h"

#ifdef WITH_CUDA
#include "fap/kernel/cuda/cuda_kernel.h"
#endif

#ifdef WITH_HIP
#include "fap/kernel/hip/hip_kernel.h"
#endif

using std::min;

namespace fap {

// naive implemention
void floyd_CPU_path(int num, float *mat, int *path) {
    for (int k = 0; k < num; k++) {
        for (int i = 0; i < num; i++) {
            for (int j = 0; j < num; j++) {
                if (mat[i * num + j] > mat[i * num + k] + mat[k * num + j]) {
                    mat[i * num + j] = mat[i * num + k] + mat[k * num + j];
                    path[i * num + j] = path[k * num + j];
                }
            }
        }
    }
}

void floyd_path_B_CPU(float *B, int *B_path,
    const int row, const int col, float *diag, int *diag_path) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float value = B[i * col + j];
            int path = B_path[i * col + j];
            for (int k = 0; k < row; k++) {
                if (value > diag[i * row + k] + B[k * col + j]) {
                    value = diag[i * row + k] + B[k * col + j];
                    path = B_path[k * col + j];
                }
            }
            B[i * col + j] = value;
            B_path[i * col + j] = path;
        }
    }
}

void floyd_path_A_CPU(float *A, int *A_path,
    const int row, const int col, float *diag, int *diag_path) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            float value = A[i * col + j];
            int path = A_path[i * col + j];
            for (int k = 0; k < col; k++) {
                if (value > A[i * col + k] + diag[k * col + j]) {
                    value = A[i * col + k] + diag[k * col + j];
                    path = diag_path[k * col + j];
                }
            }
            A[i * col + j] = value;
            A_path[i * col + j] = path;
        }
    }
}

void floyd_min_plus_CPU(float *mat1, float *mat2, int *mat2_path,
    float *res, int *path, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float value = res[i * N + j];
            int path_tmp = path[i * N + j];
            for (int k = 0; k < K; k++) {
                if (value > mat1[i * K + k] + mat2[k * N + j]) {
                    value = mat1[i * K + k] + mat2[k * N + j];
                    path_tmp = mat2_path[k * N + j];
                }
            }
            res[i * N + j] = value;
            path[i * N + j] = path_tmp;
        }
    }
}

// floyd phase2: update the col matrix
void floyd_path_A(float *A, int *A_path,
    const int row, const int col, float *diag, int *diag_path) {
#ifdef WITH_CUDA
    floyd_path_A_Nvidia(A, A_path, row, col, diag, diag_path);
#elif defined(WITH_HIP)
    floyd_path_A_AMD(A, A_path, row, col, diag, diag_path);
#else
    floyd_path_A_CPU(A, A_path, row, col, diag, diag_path);
#endif
}

// floyd phase2: update the row matrix
void floyd_path_B(float *B, int *B_path,
    const int row, const int col, float *diag, int *diag_path) {
#ifdef WITH_CUDA
    floyd_path_B_Nvidia(B, B_path, row, col, diag, diag_path);
#elif defined(WITH_HIP)
    floyd_path_B_AMD(B, B_path, row, col, diag, diag_path);
#else
    floyd_path_B_CPU(B, B_path, row, col, diag, diag_path);
#endif
}

// floyd phase3: minplus
void floyd_min_plus(float *mat1, float *mat2, int *mat2_path,
    float *res, int *path, int M, int N, int K) {
#ifdef WITH_CUDA
    floyd_min_plus_Nvidia(mat1, mat2, mat2_path, res, path, M, N, K);
#elif defined(WITH_HIP)
    floyd_min_plus_AMD(mat1, mat2, mat2_path, res, path, M, N, K);
#else
    floyd_min_plus_CPU(mat1, mat2, mat2_path, res, path, M, N, K);
#endif
}

void floyd_minplus_partition(float *mat1, float *mat2, int *mat2_path,
    float *res, int *path, int M, int N, int K, int part_num) {
#ifdef WITH_HIP
    floyd_minplus_partition_AMD(mat1, mat2, mat2_path,
        res, path, M, N, K, part_num);
#endif
}

// compute the floyd
void floyd_path(int num, float *mat, int *path) {
#ifdef WITH_CUDA
    floyd_GPU_Nvidia_path(num, mat, path);
#elif defined(WITH_HIP)
    floyd_AMD_path(num, mat, path);
#else
    floyd_CPU_path(num, mat, path);
#endif
}

}  // namespace fap

#endif
