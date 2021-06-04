#ifndef UTIL_DECENTRALIZED_H_
#define UTIL_DECENTRALIZED_H_

#include <vector>
#include <algorithm>
#include <map>
#include <unordered_set>
#include "util_centralized.h"
#include "debug.h"
#include <omp.h>

using std::cout;
using std::fill_n;
using std::string;
using std::vector;

//-----------------step3----------------------
void diagBdyMessageBuild(int *C_diagBdy_num, int *C_diagBdy_offset, int K, int *C_BlockBdy_num, int &All_diagBdy_num)
{
    int offset = 0;
    for (int i = 1; i <= K; i++)
    {
        C_diagBdy_offset[i] = offset;
        int num = C_BlockBdy_num[i];
        C_diagBdy_num[i] = num * num;
        offset += num * num;
    }
    All_diagBdy_num = offset;
}

void subDiagMatBuild(float *subDiagMat, float *subMat, int bdy_num, int sub_vertexs)
{
    for (int i = 0; i < bdy_num; i++)
    {
        for (int j = 0; j < bdy_num; j++)
        {
            subDiagMat[i * bdy_num + j] = subMat[i * sub_vertexs + j];
        }
    }
}

void add_edge(int a, int b, float c, vector<int> &h, vector<int> &e, vector<int> &ne, vector<float> &w, int &idx)
{
    e.push_back(b);
    w.push_back(c);
    ne.push_back(h[a]);
    h[a] = idx++;
}

void BG_build_solver_path(int *adj_size, int *row_offset, int *col_val, float *weight,
                          float *All_diagBdy, int bdy_vertexs, int *C_diagBdy_num, int *C_diagBdy_offset, int K,
                          int *C_BlockBdy_num, int *C_BlockBdy_offset, int *st2ed,
                          float *bdyMat, int *bdyMat_path, int myProcess, int *graph_id)
{
    //build the mapping data
    std::set<int> boundry;
    std::unordered_map<int, int> Graph2BG;
    int cnt = 0;
    for (int i = 1; i <= K; i++)
    {
        int offset = C_BlockBdy_offset[i];
        int num = C_BlockBdy_num[i];
        for (int j = 0; j < num; j++)
        {
            int ver = st2ed[j + offset];
            if (boundry.find(ver) != boundry.end())
                continue;
            boundry.insert(ver);
            Graph2BG[ver] = cnt++;
        }
    }

    vector<int> h(bdy_vertexs, -1);
    vector<int> e, ne;
    vector<float> w;
    int idx = 0;

    //add edge in subGraph
    for (int i = 1; i <= K; i++)
    {
        int offset = C_diagBdy_offset[i];
        int num = C_diagBdy_num[i];
        for (int j = 0; j < num; j++)
        {
            int ver = st2ed[j / C_BlockBdy_num[i] + C_BlockBdy_offset[i]];
            int neighbor = st2ed[j % C_BlockBdy_num[i] + C_BlockBdy_offset[i]];
            ver = Graph2BG[ver];
            neighbor = Graph2BG[neighbor];
            float wei = All_diagBdy[j + offset];
            if (wei == 0)
                continue;
            add_edge(ver, neighbor, wei, h, e, ne, w, idx);
        }
    }

    //add edges between different graph
    for (auto it = boundry.begin(); it != boundry.end(); it++)
    {
        int ver = *it;
        int adjcount = adj_size[ver];
        int offset = row_offset[ver];
        for (int i = 0; i < adjcount; i++)
        {
            int neighbor = col_val[offset + i];
            float wei = weight[offset + i];
            if (boundry.find(neighbor) == boundry.end())
                continue;
            if (graph_id[neighbor] == graph_id[ver])
                continue;
            int BG_ver = Graph2BG[ver];
            int BG_neighbor = Graph2BG[neighbor];
            add_edge(BG_ver, BG_neighbor, wei, h, e, ne, w, idx);
        }
    }

    int BG_vertexs = bdy_vertexs;
    int BG_edges = idx;
    int *BG_adj_size = (int *)malloc((BG_vertexs + 10) * sizeof(int));
    int *BG_row_offset = (int *)malloc((BG_vertexs + 10) * sizeof(int));
    int *BG_col_val = (int *)malloc((BG_edges + 10) * sizeof(int));
    float *BG_weight = (float *)malloc((BG_edges + 10) * sizeof(float));

    BG_row_offset[0] = 0;
    int edges = 0;

    for (int i = 0; i < BG_vertexs; i++)
    {
        int count = 0;
        for (int j = h[i]; j != -1; j = ne[j])
        {
            count++;
            int nextNode = e[j];
            float nextWeight = w[j];
            BG_col_val[edges] = nextNode;
            BG_weight[edges] = nextWeight;
            edges++;
        }
        BG_adj_size[i] = count;
        BG_row_offset[i + 1] = BG_row_offset[i] + count;
    }

    int source_node_num = C_BlockBdy_num[myProcess];
    int *source_node = new int[source_node_num];
    for (int i = 0; i < source_node_num; i++)
    {
        int offset = C_BlockBdy_offset[myProcess];
        int ver = st2ed[i + offset];
        ver = Graph2BG[ver];
        source_node[i] = ver;
    }
    batched_sssp_path(source_node, source_node_num, BG_vertexs, BG_edges,
                      BG_adj_size, BG_row_offset, BG_col_val, BG_weight, bdyMat, bdyMat_path);

    free(BG_adj_size);
    free(BG_row_offset);
    free(BG_col_val);
    free(BG_weight);
}

void bdy2globalBuild(std::unordered_map<int, int> &bdy2global, const int K,
                     int *C_BlockBdy_num, int *C_BlockBdy_offset, int *st2ed)
{
    int cnt = 0;
    for (int i = 1; i <= K; i++)
    {
        int offset = C_BlockBdy_offset[i];
        int num = C_BlockBdy_num[i];
        for (int j = 0; j < num; j++)
        {
            int ver = st2ed[offset + j];
            bdy2global[cnt++] = ver;
        }
    }
}

void bdyMat2global(int *bdyMat_path, const int subBdy_vertexs, const int bdy_vertexs,
                   std::unordered_map<int, int> &bdy2global)
{
    for (int i = 0; i < subBdy_vertexs; i++)
    {
        for (int j = 0; j < bdy_vertexs; j++)
        {
            int tmp = bdyMat_path[i * bdy_vertexs + j];
            if (tmp == -1)
                continue;
            bdyMat_path[i * bdy_vertexs + j] = bdy2global[tmp];
        }
    }
}

#endif
