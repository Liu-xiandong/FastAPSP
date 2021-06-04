#ifndef UTIL_READMTXFILE_H_
#define UTIL_READMTXFILE_H_

#include <bits/stdc++.h>
#include "sssp.h"

//#define DEBUG

using std::ifstream;
using std::string;

void add(int a, int b, float c,
         int *h, int *e, int *ne, float *w, int &idx)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

void readVerEdges(int &n, int &m, std::string &file, bool directed, bool weighted)
{
    std::ifstream input;

#ifdef DEBUG
    input.open("../test/mytest.txt");
#else
    if (!directed && !weighted)
    {
        input.open("../graph/unweight-undirected/" + file + ".mtx");
    }
    else if (!directed && weighted)
    {
        input.open("../graph/weight-undirected/" + file + ".mtx");
    }
    else if (directed && !weighted)
    {
        input.open("../graph/unweight-directed/" + file + ".mtx");
    }
    else
    {
        input.open("../graph/weight-directed/" + file + ".mtx");
    }
#endif

    while (input.peek() == '%')
        input.ignore(2048, '\n');

    int t;
    input >> n >> t >> m;
    if (!directed)
    {
        m = 2 * m;
    }

    input.close();
}

void readMatFile(int n, int m, int *adj_size,
                 int *row_offset, int *col_val, float *weight,
                 string &file, bool directed, bool weighted)
{
    ifstream input;
#ifdef DEBUG
    input.open("../test/mytest.txt");
#else
    if (!directed && !weighted)
    {
        input.open("../graph/unweight-undirected/" + file + ".mtx");
    }
    else if (!directed && weighted)
    {
        input.open("../graph/weight-undirected/" + file + ".mtx");
    }
    else if (directed && !weighted)
    {
        input.open("../graph/unweight-directed/" + file + ".mtx");
    }
    else
    {
        input.open("../graph/weight-directed/" + file + ".mtx");
    }
#endif

    while (input.peek() == '%')
        input.ignore(2048, '\n');

    int t;
    input >> n >> t >> m;
    if (!directed)
    {
        m = 2 * m;
    }
    int *h = (int *)malloc((n + 10) * sizeof(int));
    memset(h, -1, sizeof(int) * (n + 10));
    int *e = (int *)malloc((m + 10) * sizeof(int));
    int *ne = (int *)malloc((m + 10) * sizeof(int));
    float *w = (float *)malloc((m + 10) * sizeof(float));
    int idx = 0;

    int a, b;
    double c;
    srand((int)time(0));
#ifdef DEBUG
    while (input >> a >> b >> c)
    {
        a;
        b;
        float tc = static_cast<float>(c);
        add(a, b, tc, h, e, ne, w, idx);
        if (!directed)
        {
            add(b, a, tc, h, e, ne, w, idx);
        }
    }
#else
    if (!directed && !weighted)
    {
        while (input >> a >> b)
        {
            a--;
            b--;
            float tc = 1.0;
            // if(a == b){
            //     printf("one edge point to itself\n");
            // }
            add(a, b, tc, h, e, ne, w, idx);
            add(b, a, tc, h, e, ne, w, idx);
        }
    }
    else if (!directed && weighted)
    {
        while (input >> a >> b >> c)
        {
            a--;
            b--;
            // if(a == b){
            //     printf("one edge point to itself\n");
            // }
            c = fabs(c);
            float tc = static_cast<float>(c);
            add(a, b, tc, h, e, ne, w, idx);
            add(b, a, tc, h, e, ne, w, idx);
        }
    }
    else if (directed && !weighted)
    {
        while (input >> a >> b)
        {
            a--;
            b--;
            // if(a == b){
            //     printf("one edge point to itself\n");
            // }
            float tc = 1.0;
            add(a, b, tc, h, e, ne, w, idx);
        }
    }
    else
    {
        while (input >> a >> b >> c)
        {
            a--;
            b--;
            // if(a == b){
            //     printf("one edge point to itself\n");
            // }
            c = fabs(c);
            float tc = static_cast<float>(c);
            add(a, b, tc, h, e, ne, w, idx);
        }
    }
#endif

    row_offset[0] = 0;
    int edges = 0;

    for (int i = 0; i < n; i++)
    {
        int count = 0;
        for (int j = h[i]; j != -1; j = ne[j])
        {
            count++;
            int nextNode = e[j];
            float nextWeight = w[j];
            col_val[edges] = nextNode;
            weight[edges] = nextWeight;
            edges++;
        }
        adj_size[i] = count;
        row_offset[i + 1] = row_offset[i] + count;
    }

    if (weighted)
    {
        // bool have_cycle = negtive_cycle(n, m, adj_size, row_offset, col_val, weight);
        // assert(have_cycle == false);

        // std::cout << "johnson_reweight" << std::endl;
        // johnson_reweight(n, m, adj_size, row_offset, col_val, weight);
        // std::cout << "finish johnson_reweight" << std::endl;
        // for (int i = 0; i < m; i++)
        // {
        //     if (weight[i] < 0)
        //     {
        //         std::cout << "johnson_reweight error" << std::endl;
        //         break;
        //     }
        // }
    }

    input.close();
    free(h);
    free(e);
    free(ne);
    free(w);
}

void readMatFile_test(int &n, int &m, int *adj_size,
                      int *row_ofset, int *col_val, float *weight)
{
    //TODO
}

#endif