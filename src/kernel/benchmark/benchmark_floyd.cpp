#include "floyd.h"
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
    for (int i = 0; i < n * n; i++)
    {
        if (res[i] != mat[i]){
            printf("%f %f \n",res[i],mat[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const int n = 1024;
    float *mat = new float[n * n];
    fill_n(mat, n * n, INF);
    int m = 65536;
    time_t t;
    srand((unsigned)time(&t));
    while (m--)
    {
        int ver = rand() % (n * n) + 1;
        int i = ver / n, j = ver % n;
        mat[i * n + j] = ver % 5 + 1;
    }
    for (int i = 0; i < n; i++)
    {
        mat[i * n + i] = 0;
    }
    float *res = new float[n * n];
    memcpy(res, mat, n * n * sizeof(float));

    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    floyd_NVIDIA_GPU(n, mat);
    gettimeofday(&end, NULL);
    float elapsedTime = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("%lf s\n", elapsedTime);

    // floyd_CPU(n, res);
    // debug_matrix(mat,n,n);
    if (check(res, mat, n))
        cout << "the floyd ans is right" << endl;
    else
        cout << "the floyd ans is wrong" << endl;
}