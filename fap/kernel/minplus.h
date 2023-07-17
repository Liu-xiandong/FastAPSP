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

#ifndef MATMUL_H_
#define MATMUL_H_

#pragma once

#include <algorithm>
#include "fap/utils/parameter.h"

#ifdef WITH_CUDA
#include "fap/kernel/cuda/cuda_kernel.h"
#endif

#ifdef WITH_HIP
#include "fap/kernel/hip/hip_kernel.h"
#endif

using std::min;

namespace fap {

void min_plus_path_CPU(float *mat1, float *mat2, float *res, int *path,
    int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            res[i * N + j] = fap::MAXVALUE;
            path[i * N + j] = -1;
            for (int k = 0; k < K; k++) {
                if (res[i * N + j] > mat1[i * K + k] + mat2[k * N + j]) {
                    res[i * N + j] = mat1[i * K + k] + mat2[k * N + j];
                    path[i * N + j] = k;
                }
            }
        }
    }
}

void min_plus_path_CPU_advanced(float *mat1, float *mat2, int *mat2_path,
    float *res, int *path, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            res[i * N + j] = fap::MAXVALUE;
            path[i * N + j] = -1;
            for (int k = 0; k < K; k++) {
                if (res[i * N + j] > mat1[i * K + k] + mat2[k * N + j]) {
                    res[i * N + j] = mat1[i * K + k] + mat2[k * N + j];
                    path[i * N + j] = mat2_path[k * N + j];
                }
            }
        }
    }
}

void min_plus_path(float *mat1, float *mat2, float *res, int *path,
    int M, int N, int K) {
#ifdef WITH_CUDA
    minplus_NVIDIA_GPU(mat1, mat2, res, M, N, K);
#elif defined(WITH_HIP)
    minplus_AMD_GPU(mat1, mat2, res, M, N, K);
#else
    min_plus_path_CPU(mat1, mat2, res, path, M, N, K);
#endif
}

void min_plus_path_advanced(float *mat1, float *mat2, int *mat2_path,
    float *res, int *path, int M, int N, int K) {
#ifdef WITH_CUDA
    std::cout<<"lxd_debug: run the cuda kernel!"<<std::endl;
    minplus_NVIDIA_path(mat1, mat2, mat2_path, res, path, M, N, K);
#elif defined(WITH_HIP)
    minplus_AMD_path(mat1, mat2, mat2_path, res, path, M, N, K);
#else
    std::cout<<"lxd_debug: run the cpu kernel!"<<std::endl;
    min_plus_path_CPU_advanced(mat1, mat2, mat2_path, res, path, M, N, K);
#endif
}

}  // namespace fap

#endif