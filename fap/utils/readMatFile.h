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

#ifndef UTIL_READMTXFILE_H_
#define UTIL_READMTXFILE_H_

#pragma once

#include <bits/stdc++.h>
#include <string>

//#define DEBUG

namespace fap {

using std::ifstream;
using std::string;

void add(int a, int b, float c,
         int *h, int *e, int *ne, float *w, int &idx)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

void readVerEdges(int &n, int &m, const std::string &file,
    bool directed, bool weighted) {
    std::ifstream input;

#ifdef DEBUG
    input.open("../test/mytest.txt");
#else
    if (!directed && !weighted) {
        input.open("graph/unweight-undirected/" + file + ".mtx");
    } else if (!directed && weighted) {
        input.open("graph/weight-undirected/" + file + ".mtx");
    } else if (directed && !weighted) {
        input.open("graph/unweight-directed/" + file + ".mtx");
    } else {
        input.open("graph/weight-directed/" + file + ".mtx");
    }
#endif

    while (input.peek() == '%')
        input.ignore(2048, '\n');

    int t;
    input >> n >> t >> m;
    if (!directed) {
        m = 2 * m;
    }

    input.close();
}

void readMatFile(int n, int m, int *adj_size,
                 int *row_offset, int *col_val, float *weight,
                 const string &file, const bool directed, const bool weighted) {
    ifstream input;
#ifdef DEBUG
    input.open("../test/mytest.txt");
#else
    if (!directed && !weighted) {
        input.open("graph/unweight-undirected/" + file + ".mtx");
    } else if (!directed && weighted) {
        input.open("graph/weight-undirected/" + file + ".mtx");
    } else if (directed && !weighted) {
        input.open("graph/unweight-directed/" + file + ".mtx");
    } else {
        input.open("graph/weight-directed/" + file + ".mtx");
    }
#endif

    while (input.peek() == '%')
        input.ignore(2048, '\n');

    int t;
    input >> n >> t >> m;
    if (!directed) {
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
#ifdef DEBUG
    while (input >> a >> b >> c) {
        a;
        b;
        float tc = static_cast<float>(c);
        add(a, b, tc, h, e, ne, w, idx);
        if (!directed) {
            add(b, a, tc, h, e, ne, w, idx);
        }
    }
#else
    if (!directed && !weighted) {
        while (input >> a >> b) {
            a--;
            b--;
            float tc = 1.0;
            add(a, b, tc, h, e, ne, w, idx);
            add(b, a, tc, h, e, ne, w, idx);
        }
    } else if (!directed && weighted) {
        while (input >> a >> b >> c) {
            a--;
            b--;
            c = fabs(c);
            float tc = static_cast<float>(c);
            add(a, b, tc, h, e, ne, w, idx);
            add(b, a, tc, h, e, ne, w, idx);
        }
    } else if (directed && !weighted) {
        while (input >> a >> b) {
            a--;
            b--;
            float tc = 1.0;
            add(a, b, tc, h, e, ne, w, idx);
        }
    } else {
        while (input >> a >> b >> c) {
            a--;
            b--;
            c = fabs(c);
            float tc = static_cast<float>(c);
            add(a, b, tc, h, e, ne, w, idx);
        }
    }
#endif

    row_offset[0] = 0;
    int edges = 0;

    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = h[i]; j != -1; j = ne[j]) {
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

    if (weighted) {
        // TODO(Liu-xiandong): Handle cases with negative weighted.
    }

    input.close();
    free(h);
    free(e);
    free(ne);
    free(w);
}

}  // namespace fap

#endif  // UTIL_READMTXFILE_H_
