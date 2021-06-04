#ifndef UTIL_SSSP_H_
#define UTIL_SSSP_H_

#include <algorithm>
#include <string.h>
#include <queue>
#include <vector>
#include <omp.h>
#include "debug.h"
#include "parameter.h"

//#define USE_CPU

// using namespace boost;
using std::fill_n;
using std::vector;

extern "C" void batched_sssp_cuGraph(int *source_node, int source_node_num, int vertexs, int edges,
									 int *adj_size, int *row_offset, int *col_val, float *weights,
									 float *batched_dist, int *batched_path);

extern "C" void handle_boundry_AMD_GPU(float *subGraph, int vertexs, int edges, int bdy_num,
									   int *adj_size, int *row_offset, int *col_val, float *weight,
									   int *st2ed, int offset);

void dijkstra(int u, int vertexs, int *adj_size, int *row_offset, int *col_val, float *weight, float *dist)
{
	fill_n(dist, vertexs, MAXVALUE);
	dist[u] = 0;

	std::vector<bool> st(vertexs, false);

	typedef std::pair<float, int> PDI;
	std::priority_queue<PDI, std::vector<PDI>, std::greater<PDI>> q;
	q.push({0, u});

	while (!q.empty())
	{
		auto x = q.top();
		q.pop();
		int ver = x.second;
		float distance = x.first;

		if (st[ver])
			continue;
		st[ver] = true;

		int adjcount = adj_size[ver];
		int offset = row_offset[ver];
		for (int i = 0; i < adjcount; i++)
		{
			int nextNode = col_val[offset + i];
			float w = weight[offset + i];
			if (dist[nextNode] > distance + w)
			{
				q.push({distance + w, nextNode});
				dist[nextNode] = distance + w;
			}
		}
	}
}

void dijkstra_path(int u, int vertexs, int *adj_size, int *row_offset, int *col_val, float *weight, float *dist, int *path)
{
	fill_n(dist, vertexs, MAXVALUE);
	fill_n(path, vertexs, -1);
	dist[u] = 0;
	path[u] = u;

	std::vector<bool> st(vertexs, false);

	typedef std::pair<float, int> PDI;
	std::priority_queue<PDI, std::vector<PDI>, std::greater<PDI>> q;
	q.push({0, u});

	while (!q.empty())
	{
		auto x = q.top();
		q.pop();
		int ver = x.second;
		float distance = x.first;

		if (st[ver])
			continue;
		st[ver] = true;

		int adjcount = adj_size[ver];
		int offset = row_offset[ver];
		for (int i = 0; i < adjcount; i++)
		{
			int nextNode = col_val[offset + i];
			float w = weight[offset + i];
			if (dist[nextNode] > distance + w)
			{
				q.push({distance + w, nextNode});
				dist[nextNode] = distance + w;
				path[nextNode] = ver;
			}
		}
	}
}

struct Edge
{
	int a, b;
	float c;
};

void johnson_spfa(int u, int vertexs, int edges, int *adj_size, int *row_offset, int *col_val, float *weight, float *dist)
{
	vector<int> adj_size_reweight(vertexs + 1);
	vector<int> row_offset_reweight(vertexs + 2);
	vector<int> col_val_reweight(edges + vertexs);
	vector<float> weight_reweight(edges + vertexs);

	//add edges
	memmove((int *)&adj_size_reweight[0], adj_size, vertexs * sizeof(int));
	adj_size_reweight[vertexs] = vertexs;
	//row_offset
	memmove((int *)&row_offset_reweight[0], row_offset, (vertexs + 1) * sizeof(int));
	row_offset_reweight[vertexs + 1] = row_offset_reweight[vertexs] + vertexs;
	//col and weight
	memmove((int *)&col_val_reweight[0], col_val, edges * sizeof(int));
	for (int i = 0; i < vertexs; i++)
	{
		col_val_reweight[edges + i] = i;
	}
	memmove((float *)&weight_reweight[0], weight, edges * sizeof(float));
	for (int i = 0; i < vertexs; i++)
	{
		weight_reweight[edges + i] = 0;
	}

	vector<Edge> edge_tmp(edges + vertexs);
	int cnt = 0;
	for (int i = 0; i < vertexs + 1; i++)
	{
		int adjcount = adj_size_reweight[i];
		int offset = row_offset_reweight[i];
		for (int j = 0; j < adjcount; j++)
		{
			int nextNode = col_val_reweight[offset + j];
			float w = weight_reweight[offset + j];
			edge_tmp[cnt].a = i;
			edge_tmp[cnt].b = nextNode;
			edge_tmp[cnt].c = w;
			cnt++;
		}
	}

	//spfa
	fill_n(dist, vertexs + 1, MAXVALUE);
	dist[vertexs] = 0;

	vector<float> last(vertexs + 1);

	for (int i = 0; i < vertexs + 1; i++)
	{
		memcpy(last.data(), dist, (vertexs + 1) * sizeof(float));
		for (int j = 0; j < edges + vertexs; j++)
		{
			auto e = edge_tmp[j];
			if (dist[e.b] > last[e.a] + e.c)
			{
				dist[e.b] = last[e.a] + e.c;
			}
		}
	}

	return;
}

void johnson_reweight(const int vertexs, const int edges,
					  int *adj_size, int *row_offset, int *col_val, float *weights)
{
	vector<float> modify_dist(vertexs + 1);

	johnson_spfa(vertexs, vertexs, edges, adj_size, row_offset, col_val, weights, (float *)&modify_dist[0]);

	//debug_array(modify_dist.data(), vertexs + 1);
	// re-weight
	for (int i = 0; i < vertexs; i++)
	{
		int adjcount = adj_size[i];
		int offset = row_offset[i];
		for (int j = 0; j < adjcount; j++)
		{
			int nextNode = col_val[offset + j];
			weights[offset + j] = weights[offset + j] + modify_dist[i] - modify_dist[nextNode];
		}
	}
	//debug_array(weights, edges);
}

bool negtive_cycle(const int vertexs, const int edges,
				   int *adj_size, int *row_offset, int *col_val, float *weights)
{
	vector<float> dist(vertexs, 0);
	std::queue<int> q;
	std::vector<bool> st(vertexs, false);
	for (int i = 0; i < vertexs; i++)
	{
		q.push(i);
		st[i] = true;
	}
	vector<int> cnt(vertexs, 0);

	while (!q.empty())
	{
		int t = q.front();
		q.pop();
		st[t] = false;

		int adjcount = adj_size[t];
		int offset = row_offset[t];
		for (int i = 0; i < adjcount; i++)
		{
			int nextNode = col_val[offset + i];
			float w = weights[offset + i];
			if (dist[nextNode] > dist[t] + w)
			{
				dist[nextNode] = dist[t] + w;
				cnt[nextNode] = cnt[t] + 1;
				if (cnt[nextNode] >= vertexs)
					return true;
				if (!st[nextNode])
				{
					st[nextNode] = true;
					q.push(nextNode);
				}
			}
		}
	}
	return false;
}

void batched_sssp_path(int *source_node, int source_node_num, int vertexs, int edges,
					   int *adj_size, int *row_offset, int *col_val, float *weights, float *batched_dist, int *batched_path)
{
#ifdef USE_CPU
	#pragma omp parallel
	{
	#pragma omp for
		for (int i = 0; i < source_node_num; i++)
		{
			int ver = source_node[i];
			dijkstra_path(ver, vertexs, adj_size, row_offset, col_val, weights,
						  batched_dist + (long long)i * vertexs,
						  batched_path + (long long)i * vertexs);
		}
	}
#else
	batched_sssp_cuGraph(source_node, source_node_num, vertexs, edges,
						 adj_size, row_offset, col_val, weights,
						 batched_dist, batched_path);
#endif
}

//TODO
void handle_boundry_path(float *subGraph, int *subGraph_path, int vertexs, int edges, int bdy_num,
						 int *adj_size, int *row_offset, int *col_val, float *weights,
						 int *st2ed, int offset)
{
	// #pragma omp parallel
	// 	{
	// #pragma omp for
	// 		for (int i = 0; i < bdy_num; i++)
	// 		{
	// 			int ver = st2ed[offset + i];
	// 			dijkstra_path(ver, vertexs, adj_size, row_offset, col_val, weights,
	// 						  subGraph + (long long)i * vertexs, subGraph_path + (long long)i * vertexs);
	// 		}
	// 	}

	int *source_node = new int[bdy_num];
	for (int i = 0; i < bdy_num; i++)
	{
		source_node[i] = st2ed[offset + i];
	}
	batched_sssp_path(source_node, bdy_num, vertexs, edges,
					  adj_size, row_offset, col_val, weights, subGraph, subGraph_path);
}

#endif
