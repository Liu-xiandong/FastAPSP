#include "MatMul.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <iostream>
#include "debug.h"

using namespace std;
#define INF 1e8

bool check(float *res, float *mat, const int n)
{
    for (int i = 0; i < n; i++)
    {
        if (res[i] != mat[i])
            return false;
    }
    return true;
}

int main()
{
    const int m = 36, n = 36, k = 36;
    float *a = new float[m * k];
    float *b = new float[k * n];
    float *c = new float[m * n];
    float *res = new float[m * n];

    fill_n(c, m * n, INF);
    fill_n(res, m * n, INF);

    for (int i = 0; i < m * k; i++)
    {
        a[i] = i / 1024;
    }
    for (int i = 0; i < k * n; i++)
    {
        b[i] = i % 512;
    }

    time_t t;
    srand((unsigned)time(&t));
    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    gemm_AMD_GPU(a, b, c, m, n, k);
    gettimeofday(&end, NULL);
    float elapsedTime = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("the time is %lf s\n", elapsedTime);

    // gemm_CPU(a, b, res, m, n, k);
    // debug_matrix(res,m,n);
    // printf("________________________________\n");
    // debug_matrix(c,m,n);
    // if (check(res, c, m * n))
    //     cout << "the gemm ans is right" << endl;
    // else
    //     cout << "the gemm ans is wrong" << endl;

}