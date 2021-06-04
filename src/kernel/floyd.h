#ifndef FLOYD_H_
#define FLOYD_H_
#include <algorithm>
#include "parameter.h"
#include <iostream>

using std::min;

#define DEVICE_NUM 2

extern "C" void floyd_NVIDIA_GPU(int num_node, float *arc);
extern "C" void floyd_AMD_GPU(int num_node, float *arc);
extern "C" void floyd_GPU_Nvidia_path(int num_node, float *arc, int *path);
extern "C" void floyd_AMD_path(int num_node, float *arc, int *path);

extern "C" void floyd_path_A_Nvidia(float *A, int *A_path, const int row, const int col, float *diag, int *diag_path);
extern "C" void floyd_path_B_Nvidia(float *B, int *B_path, const int row, const int col, float *diag, int *diag_path);
extern "C" void floyd_min_plus_Nvidia(float *mat1, float *mat2, int *mat2_path, float *res, int *res_path, int m, int n, int k);
extern "C" void floyd_path_A_AMD(float *A, int *A_path, const int row, const int col, float *diag, int *diag_path);
extern "C" void floyd_path_B_AMD(float *B, int *B_path, const int row, const int col, float *diag, int *diag_path);
extern "C" void floyd_min_plus_AMD(float *mat1, float *mat2, int *mat2_path, float *res, int *res_path, int m, int n, int k);

extern "C" void floyd_minplus_partition_AMD(float *mat1, float *mat2, int *mat2_path,
                                            float *res, int *res_path, int m, int n, int k, const int part_num);

void floyd_CPU_path(int num, float *mat, int *path)
{
    for (int k = 0; k < num; k++)
    {
        for (int i = 0; i < num; i++)
        {
            for (int j = 0; j < num; j++)
            {
                if (mat[i * num + j] > mat[i * num + k] + mat[k * num + j])
                {
                    mat[i * num + j] = mat[i * num + k] + mat[k * num + j];
                    path[i * num + j] = path[k * num + j];
                }
            }
        }
    }
}

void floyd_path_B_CPU(float *B, int *B_path, const int row, const int col, float *diag, int *diag_path)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            float value = B[i * col + j];
            int path = B_path[i * col + j];
            for (int k = 0; k < row; k++)
            {
                if (value > diag[i * row + k] + B[k * col + j])
                {
                    value = diag[i * row + k] + B[k * col + j];
                    path = B_path[k * col + j];
                }
            }
            B[i * col + j] = value;
            B_path[i * col + j] = path;
        }
    }
}

void floyd_path_A_CPU(float *A, int *A_path, const int row, const int col, float *diag, int *diag_path)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            float value = A[i * col + j];
            int path = A_path[i * col + j];
            for (int k = 0; k < col; k++)
            {
                if (value > A[i * col + k] + diag[k * col + j])
                {
                    value = A[i * col + k] + diag[k * col + j];
                    path = diag_path[k * col + j];
                }
            }
            A[i * col + j] = value;
            A_path[i * col + j] = path;
        }
    }
}

void floyd_min_plus_CPU(float *mat1, float *mat2, int *mat2_path, float *res, int *path, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float value = res[i * N + j];
            int path_tmp = path[i * N + j];
            for (int k = 0; k < K; k++)
            {
                if (value > mat1[i * K + k] + mat2[k * N + j])
                {
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
void floyd_path_A(float *A, int *A_path, const int row, const int col, float *diag, int *diag_path)
{
#if DEVICE_NUM == 0
    floyd_path_A_CPU(A, A_path, row, col, diag, diag_path);
#elif DEVICE_NUM == 1
    floyd_path_A_Nvidia(A, A_path, row, col, diag, diag_path);
#else
    floyd_path_A_AMD(A, A_path, row, col, diag, diag_path);
#endif
}

// floyd phase2: update the row matrix
void floyd_path_B(float *B, int *B_path, const int row, const int col, float *diag, int *diag_path)
{
#if DEVICE_NUM == 0
    {
        floyd_path_B_CPU(B, B_path, row, col, diag, diag_path);
    }
#elif DEVICE_NUM == 1
    {
        floyd_path_B_Nvidia(B, B_path, row, col, diag, diag_path);
    }
#else
    {
        floyd_path_B_AMD(B, B_path, row, col, diag, diag_path);
    }
#endif
}

// floyd phase3: minplus
void floyd_min_plus(float *mat1, float *mat2, int *mat2_path, float *res, int *path, int M, int N, int K)
{
#if DEVICE_NUM == 0
    {
        floyd_min_plus_CPU(mat1, mat2, mat2_path, res, path, M, N, K);
    }
#elif DEVICE_NUM == 1
    {
        floyd_min_plus_Nvidia(mat1, mat2, mat2_path, res, path, M, N, K);
    }
#else
    {
        floyd_min_plus_AMD(mat1, mat2, mat2_path, res, path, M, N, K);
    }
#endif
}

void floyd_minplus_partition(float *mat1, float *mat2, int *mat2_path, float *res, int *path, int M, int N, int K, int part_num)
{
    floyd_minplus_partition_AMD(mat1, mat2, mat2_path, res, path, M, N, K, part_num);
}

// compute the floyd
void floyd_path(int num, float *mat, int *path)
{
    // #if DEVICE_NUM == 0
    // {
    //     floyd_CPU_path(num, mat, path);
    // }
    // #elif DEVICE_NUM == 1
    // {
    //     floyd_GPU_Nvidia_path(num, mat, path);
    // }
    // #else
    // {

    floyd_AMD_path(num, mat, path);

    // }
    // #endif
}
#endif