#include <bits/stdc++.h>
#include "sssp.h"
#include "readMatFile.h"
#include "readIdFile.h"
#include <time.h>
#include <sys/time.h>
#include <omp.h>

//hip RunTime API

using namespace std;

extern "C" void multi_source_Nvidia_sssp(int numFrom, int *fromNode, int vertexs, int edges, int *rowOffsetArc, int *colValueArc, float *weightArc, float *shortLenTable);

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
    string file = "road-usroads";
    int vertexs;
    int edges;
    readVerEdges(vertexs, edges, file, false, false);
    cout << vertexs << " " << edges << endl;

    int *adj_size = (int *)malloc(vertexs * sizeof(int));
    int *row_offset = (int *)malloc(vertexs * sizeof(int));
    int *col_val = (int *)malloc(edges * sizeof(int));
    float *weight = (float *)malloc(edges * sizeof(float));
    readMatFile(vertexs, edges, adj_size, row_offset, col_val, weight, file, false, false);

    int source_num = 250;
    int *sources = new int[source_num];
    float *res_GPU = new float[source_num * vertexs];
    float *res_CPU = new float[source_num * vertexs];
    for (int i = 0; i < source_num; i++)
    {
        sources[i] = i;
    }

    time_t t;
    srand((unsigned)time(&t));
    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    multi_source_Nvidia_sssp(source_num, sources, vertexs, edges, row_offset, col_val, weight, res_GPU);
    gettimeofday(&end, NULL);
    float elapsedTime = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("the GPU time is: %lf s\n", elapsedTime);

    gettimeofday(&begin, NULL);
    int n = 1;
#pragma omp parallel
    {
        n = omp_get_num_threads();
        printf("the total threads is: %d\n", n);
    }

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < source_num; i++)
        {
            int ver = i;
            dijkstra(ver, vertexs, adj_size, row_offset, col_val, weight, res_CPU + (long long)i * vertexs);
        }
    }
    gettimeofday(&end, NULL);
    elapsedTime = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("the CPU time is: %lf s\n", elapsedTime);

    if (check(res_CPU, res_GPU, source_num * vertexs))
        cout << "the sssp ans is right" << endl;
    else
        cout << "the sssp ans is wrong" << endl;
}