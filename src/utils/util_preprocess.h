#ifndef UTIL_PREPROCESS_H_
#define UTIL_PREPROCESS_H_

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sssp.h"

void findBoundry(int K, int n, int *id, int *adj_size,
                 int *row_offset, int *col_val, float *weight,
                 std::unordered_map<int, std::vector<int>> &BlockVer,
                 std::unordered_map<int, std::vector<int>> &BlockBoundary,
                 std::unordered_map<int, int> &isBoundry)
{
    for (int subId = 1; subId <= K; subId++)
    {
        for (int i = 0; i < BlockVer[subId].size(); i++)
        {
            int u = BlockVer[subId][i];
            int adjcount = adj_size[u];
            int offset = row_offset[u];

            for (int j = 0; j < adjcount; j++)
            {
                int nextNode = col_val[offset + j];
                if (id[nextNode] != id[u])
                {
                    isBoundry[u] = 1;
                    isBoundry[nextNode] = 1;
                }
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        if (isBoundry[i])
        {
            BlockBoundary[id[i]].push_back(i);
        }
    }
}

struct dense_node
{
    int v;
    int GraphId;
    int isBound;
};

static bool cmp(const dense_node &a, const dense_node &b)
{
    if (a.GraphId == b.GraphId)
        return a.isBound > b.isBound;
    else
        return a.GraphId < b.GraphId;
}

void sort_and_encode(int K, int n, int *id,
                     std::unordered_map<int, int> &isBoundry,
                     int *C_BlockVer_num, int *C_BlockVer_offset,
                     int *C_BlockBdy_num, int *C_BlockBdy_offset,
                     std::unordered_map<int, std::vector<int>> &BlockVer,
                     std::unordered_map<int, std::vector<int>> &BlockBoundary,
                     int *st2ed, int *ed2st)
{
    vector<dense_node> ver;
    for (int i = 0; i < n; i++)
    {
        dense_node cur;
        cur.v = i, cur.GraphId = id[i], cur.isBound = isBoundry[i];
        ver.push_back(cur);
    }

    vector<dense_node> verSorted(ver);
    sort(verSorted.begin(), verSorted.end(), cmp);

    for (int i = 0; i < ver.size(); i++)
    {
        int before = ver[i].v;
        int after = verSorted[i].v;
        st2ed[before] = after;
        ed2st[after] = before;
    }

    //vector2pointer
    int cnt = 0;
    for (int i = 1; i <= K; i++)
    {
        C_BlockVer_offset[i] = cnt;
        C_BlockVer_num[i] = BlockVer[i].size();
        C_BlockBdy_offset[i] = cnt;
        C_BlockBdy_num[i] = BlockBoundary[i].size();
        cnt += BlockVer[i].size();
    }
}

#endif