#ifndef UTIL_READIDFILE_H_
#define UTIL_READIDFILE_H_

#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <metis.h>
#include "debug.h"

using std::ifstream;
using std::string;
using std::to_string;
using std::unordered_map;
using std::vector;

//#define DEBUG

void readIdFile(int n, int *id, std::unordered_map<int, std::vector<int>> &BlockVer,
                int K, bool directed, bool weighted)
{
    std::ifstream input;
#ifdef DEBUG
    input.open("../test/subtest.txt");

    int a, b;
    string line;
    while (input >> line)
    {
        int pre = 0, cnt = 0;
        for (int i = 0; i < line.size(); i++)
        {
            if (cnt == 0 && line[i] == ',')
            {
                a = stoi(line.substr(pre, i - pre));
                cnt++;
                pre = i + 1;
            }
            if (cnt == 1 && line[i] == ',')
            {
                b = stoi(line.substr(pre));
                break;
            }
        }
        id[a] = b;
        BlockVer[b].push_back(a);
    }
#endif
}

//if the input csr is directed graph, need to chang it to undirected graph

void Graph_decomposition(int *id, std::unordered_map<int, std::vector<int>> &BlockVer, bool directed,
                         int K, int vertexs, int edges, int *adj_size, int *row_offset, int *col_val)
{
    std::vector<idx_t> xadj(1);
    std::vector<idx_t> adjncy;
    if (directed)
    {
        //change to undirected graph
        std::map<int, std::unordered_set<int>> graph;
        for (int i = 0; i < vertexs; i++)
        {
            int ver = i;
            int adjcount = adj_size[ver];
            int offset = row_offset[ver];
            for (int i = 0; i < adjcount; i++)
            {
                int nextNode = col_val[offset + i];
                if (ver == nextNode)
                {
                    continue;
                }
                graph[ver].insert(nextNode);
                graph[nextNode].insert(ver);
            }
        }

        for (auto it = graph.begin(); it != graph.end(); it++)
        {
            auto ver = it->first;
            int count = 0;
            for (auto it = graph[ver].begin(); it != graph[ver].end(); it++)
            {
                int nextNode = *it;
                count++;
                adjncy.push_back(nextNode);
            }
            xadj.push_back(xadj.back() + count);
        }
    }
    else
    {
        std::vector<idx_t> xadj_tmp(row_offset, row_offset + vertexs + 1);
        std::vector<idx_t> adjncy_tmp(col_val, col_val + edges);
        xadj.assign(xadj_tmp.begin(), xadj_tmp.end());
        adjncy.assign(adjncy_tmp.begin(), adjncy_tmp.end());
    }
    idx_t nVertices = vertexs;
    idx_t nWeights = 1;
    idx_t nParts = K;
    idx_t objval;
    std::vector<idx_t> part(nVertices, 0);

    std::vector<idx_t> vwgt(nVertices * nWeights, 0);
    //std::cout << "METIS_PartGraphKway start" << std::endl;
    int ret = METIS_PartGraphKway(&nVertices, &nWeights, xadj.data(), adjncy.data(),
                                  NULL, NULL, NULL, &nParts, NULL,
                                  NULL, NULL, &objval, part.data());
    //std::cout << "METIS_PartGraphKway end" << std::endl;

    if (ret != rstatus_et::METIS_OK)
    {
        std::cout << "METIS_ERROR" << std::endl;
    }

    for (unsigned i = 0; i < part.size(); i++)
    {
        int subGraph_id = part[i] + 1;
        id[i] = subGraph_id;
        BlockVer[subGraph_id].push_back(i);
    }
}

void readIdFile_METIS(int *id, std::unordered_map<int, std::vector<int>> &BlockVer,
                      int K, bool directed, int vertexs, int edges, int *adj_size,
                      int *row_offset, int *col_val, float *weight)
{
#ifdef DEBUG
    readIdFile(vertexs, id, BlockVer, K, directed, true);
#else
    Graph_decomposition(id, BlockVer, directed, K, vertexs, edges, adj_size, row_offset, col_val);
#endif
    // debug_array(id,vertexs);
    // for(int i=1; i<=K; i++){
    //     auto cur = BlockVer[i];
    //     debug_array((int *)&cur[0], cur.size());
    // }
}

#endif
