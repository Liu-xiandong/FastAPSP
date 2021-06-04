#ifndef MATMUL_H_
#define MATMUL_H_

#include <algorithm>
#include "parameter.h"

using std::min;

#define DEVICE_NUM 2

extern "C" void gemm_NVIDIA_GPU(float *mat1, float *mat2, float *res, int M, int N, int K);
extern "C" void gemm_AMD_GPU(float *mat1, float *mat2, float *res, int M, int N, int K);
extern "C" void gemm_NVIDIA_path(float *mat1, float *mat2, int *mat2_path, float *res, int *res_path, int m, int n, int k);
extern "C" void gemm_NVIDIA_path_backup(float *mat1, float *mat2, int *mat2_path, float *res, int *res_path, int m, int n, int k);
extern "C" void gemm_AMD_path(float *mat1, float *mat2, int *mat2_path, float *res, int *res_path, int m, int n, int k);

extern "C" void gemm_AMD_partition(float *mat1, float *mat2, int *mat2_path,
                                   int m, int n, int k, const int part_num,
                                   float *subMat, int *subMat_path,
                                   int start, int sub_vertexs, int vertexs, int *st2ed);

// void gemm_CPU(float *mat1, float *mat2, float *res, int M, int N, int K)
// {
//     for (int i = 0; i < M; i++)
//     {
//         for (int j = 0; j < N; j++)
//         {
//             res[i * N + j] = 1e12;
//             for (int k = 0; k < K; k++)
//             {
//                 res[i * N + j] = min(res[i * N + j], mat1[i * K + k] + mat2[k * N + j]);
//             }
//         }
//     }
// }

// void gemm(float *mat1, float *mat2, float *res, int M, int N, int K)
// {
//     #if DEVICE_NUM == 0
//     {
//         gemm_CPU(mat1, mat2, res, M, N, K);
//     }
//     #elif DEVICE_NUM == 1
//     {
//         gemm_NVIDIA_GPU(mat1, mat2, res, M, N, K);
//     }
//     #else
//     {
//         gemm_AMD_GPU(mat1, mat2, res, M, N, K);
//     }
//     #endif
// }

void min_plus_path_CPU(float *mat1, float *mat2, float *res, int *path, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            res[i * N + j] = MAXVALUE;
            path[i * N + j] = -1;
            for (int k = 0; k < K; k++)
            {
                if (res[i * N + j] > mat1[i * K + k] + mat2[k * N + j])
                {
                    res[i * N + j] = mat1[i * K + k] + mat2[k * N + j];
                    path[i * N + j] = k;
                }
            }
        }
    }
}

void min_plus_path_CPU_advanced(float *mat1, float *mat2, int *mat2_path, float *res, int *path, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            res[i * N + j] = MAXVALUE;
            path[i * N + j] = -1;
            for (int k = 0; k < K; k++)
            {
                if (res[i * N + j] > mat1[i * K + k] + mat2[k * N + j])
                {
                    res[i * N + j] = mat1[i * K + k] + mat2[k * N + j];
                    path[i * N + j] = mat2_path[k * N + j];
                }
            }
        }
    }
}

void min_plus_path(float *mat1, float *mat2, float *res, int *path, int M, int N, int K)
{
    #if DEVICE_NUM == 0
    {
        min_plus_path_CPU(mat1, mat2, res, path, M, N, K);
    }
    #elif DEVICE_NUM == 1
    {
        gemm_NVIDIA_GPU(mat1, mat2, res, M, N, K);
    }
    #else
    {
        gemm_AMD_GPU(mat1, mat2, res, M, N, K);
    }
    #endif
}

void min_plus_path_advanced(float *mat1, float *mat2, int *mat2_path, float *res, int *path, int M, int N, int K)
{
    #if DEVICE_NUM == 0
    {
        min_plus_path_CPU_advanced(mat1, mat2, mat2_path, res, path, M, N, K);
    }
    #elif DEVICE_NUM == 1
    {
        gemm_NVIDIA_path(mat1, mat2, mat2_path, res, path, M, N, K);
    }
    #else
    {
        gemm_AMD_path(mat1, mat2, mat2_path, res, path, M, N, K);
    }
    #endif
}

#endif