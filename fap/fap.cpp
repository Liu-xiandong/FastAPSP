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

#include <assert.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "fap/fap.h"
#include "fap/utils.h"
#include "fap/kernel.h"

namespace fap {

fapGraph::fapGraph(std::string input_graph,
                  bool directed, bool weighted, int32_t K) {
  // build graph from file.
  int vertexs;
  int edges;

  readVerEdges(vertexs, edges, input_graph, directed, weighted);
  this->num_vertexs = vertexs;
  this->num_edges = edges;
  this->K = K;

  this->adj_size.resize(vertexs);
  this->row_offset.resize(vertexs + 1);
  this->col_val.resize(edges);
  this->weight.resize(edges);

  this->graph_id.assign(vertexs, 0);
  this->st2ed.assign(vertexs, 0);
  this->ed2st.assign(vertexs, 0);
  this->C_BlockVer_num.assign(K + 1, 0);
  this->C_BlockVer_offset.assign(K + 1, 0);
  this->C_BlockBdy_num.assign(K + 1, 0);
  this->C_BlockBdy_offset.assign(K + 1, 0);

  // get message of Graph from input file.
  readMatFile(vertexs, edges,
      adj_size.data(), row_offset.data(), col_val.data(), weight.data(),
      input_graph, directed, weighted);
  // get subgraph message from metis.
  readIdFile_METIS(
      graph_id.data(), BlockVer, K, directed, vertexs, edges,
      adj_size.data(), row_offset.data(), col_val.data(), weight.data());
}

fapGraph::fapGraph(int32_t num_vertexs, int64_t num_edges,
                std::vector<int32_t> row_offset, std::vector<int32_t> col_val,
                std::vector<float> weight, int32_t K) {
  // build graph from metadata.
}

void fapGraph::preCondition() {
  findBoundry(this->K, this->num_vertexs, this->graph_id.data(),
            this->adj_size.data(), this->row_offset.data(),
            this->col_val.data(), this->weight.data(),
            this->BlockVer, this->BlockBoundary, this->isBoundry);

  sort_and_encode(this->K, this->num_vertexs,
                  this->graph_id.data(), this->isBoundry,
                  this->C_BlockVer_num.data(), this->C_BlockVer_offset.data(),
                  this->C_BlockBdy_num.data(), this->C_BlockBdy_offset.data(),
                  this->BlockVer, this->BlockBoundary,
                  this->st2ed.data(), this->ed2st.data());

  this->is_split = (this->num_vertexs > this->max_vertexs_num_limit);
}

bool fapGraph::isSplit() {
  return this->is_split;
}

std::vector<int> fapGraph::getGraphId() {
  return this->graph_id;
}

// run fast APSP algorithm.
void fapGraph::run(float *subgraph_dist,
                      int *subgraph_path,
                      const int32_t sub_garph_id) {
  const int bdy_num = C_BlockBdy_num[sub_garph_id];
  const int sub_vertexs = C_BlockVer_num[sub_garph_id];
  const int inner_num = sub_vertexs - bdy_num;
  int64_t subgraph_dist_size = (int64_t)sub_vertexs * this->num_vertexs;
  int64_t inner_to_bdy_size = (int64_t)inner_num * bdy_num;
  int64_t inner_to_inner_size = (int64_t)sub_vertexs * sub_vertexs;
  vector<float> inner_to_bdy_dist(inner_to_bdy_size, fap::MAXVALUE);
  vector<float> inner_to_inner_dist(inner_to_inner_size, fap::MAXVALUE);
  vector<int> inner_to_inner_path(inner_to_inner_size, -1);

  // stage 1. run sssp algorithm in boundry points.
  fap::handle_boundry_path(
            subgraph_dist, subgraph_path,
            this->num_vertexs, this->num_edges, bdy_num,
            adj_size.data(), row_offset.data(), col_val.data(), weight.data(),
            st2ed.data(), C_BlockVer_offset[sub_garph_id]);

  // stage 2. run floyd algorithm in all points of subgraph.
  // 2.1 move data from subgraph matrix to floyd matrix.
  fap::MysubMatBuild_path(
      inner_to_inner_dist.data(), inner_to_inner_path.data(),
      subgraph_dist, subgraph_path,
      graph_id.data(), C_BlockVer_offset[sub_garph_id],
      sub_vertexs, bdy_num,
      this->num_vertexs, st2ed.data(), ed2st.data(),
      adj_size.data(), row_offset.data(), col_val.data(), weight.data());

  // 2.2 run floyd algorithm
  fap::floyd_path(sub_vertexs,
    inner_to_inner_dist.data(), inner_to_inner_path.data());

  // stage 3. run min-plus algorithm in boundry points to all other points.
  // 3.1 move data from subMat to a thin matrix.
  fap::MyMat1Build(inner_to_bdy_dist.data(), inner_to_inner_dist.data(),
            inner_num, bdy_num, sub_vertexs);
  int64_t offset = (int64_t)bdy_num * this->num_vertexs;

  // 3.2 run min-plus
  // GPU global mem / sizeof(float)
  const double GPU_MAX_NUM = 4e9;
  const double MEM_NUM = GPU_MAX_NUM / this->num_vertexs - bdy_num;
  int part_num = 1;
#ifdef WITH_GPU
  part_num = static_cast<int>(
      ceil(static_cast<double>(inner_num) / MEM_NUM));
#endif

  // GPU memory is expansive and maybe too big.
  if (part_num == 1) {
      fap::min_plus_path_advanced(
          inner_to_bdy_dist.data(),
          subgraph_dist, subgraph_path,
          subgraph_dist + offset,
          subgraph_path + offset,
          inner_num, this->num_vertexs, bdy_num);
  } else {
      int block_size = inner_num / part_num;
      int last_size = inner_num - block_size * (part_num - 1);

      for (int i = 0; i < part_num; i++) {
        int64_t offset_value = offset +
          (int64_t)i * block_size * this->num_vertexs;
        if (i == part_num - 1) {
            fap::min_plus_path_advanced(
              inner_to_bdy_dist.data() + i * block_size * bdy_num,
              subgraph_dist, subgraph_path,
              subgraph_dist + offset_value,
              subgraph_path + offset_value,
              last_size, this->num_vertexs, bdy_num);
        } else {
            fap::min_plus_path_advanced(
              inner_to_bdy_dist.data() + i * block_size * bdy_num,
              subgraph_dist, subgraph_path,
              subgraph_dist + offset_value,
              subgraph_path + offset_value,
              block_size, this->num_vertexs, bdy_num);
        }
      }
  }

  // 3.3 move data from floyd matrix to subgraph matrix.
  fap::MysubMatDecode_path(
      inner_to_inner_dist.data(), inner_to_inner_path.data(),
      subgraph_dist, subgraph_path,
      C_BlockVer_offset[sub_garph_id], sub_vertexs,
      sub_vertexs, this->num_vertexs, st2ed.data());
}

// run fast APSP algorithm.
int32_t fapGraph::solve(bool is_path_needed) {
  int64_t graph_size = (int64_t)num_vertexs * num_vertexs;
  dist.resize(graph_size);
  path.resize(graph_size);
  int64_t offset = 0;
  for (int i = 1; i <= K; i++) {
    run(dist.data() + offset, path.data() + offset, i);
    int64_t size = (int64_t)num_vertexs * C_BlockVer_num[i];
    offset += size;
  }
}

// run fast APSP algorithm in one subgraph.
int32_t fapGraph::solveSubGraph(int32_t sub_garph_id,
                              bool is_path_needed) {
  // init data.
  const int sub_vertexs = C_BlockVer_num[sub_garph_id];
  int64_t subgraph_dist_size = (int64_t)sub_vertexs * this->num_vertexs;
  this->subgraph_dist.resize(subgraph_dist_size);
  this->subgraph_path.resize(subgraph_dist_size);
  this->current_subgraph_id = sub_garph_id;

  run(subgraph_dist.data(), subgraph_path.data(), sub_garph_id);
}

std::vector<int32_t> fapGraph::getMapping() {
  return this->st2ed;
}

std::vector<float> fapGraph::getAllDistance() {
  assert(this->is_split == false);
  return this->dist;
}

std::vector<int32_t> fapGraph::getAllPath() {
  assert(this->is_split == false);
  return this->path;
}

std::vector<int32_t> fapGraph::getSubGraphIndex(int32_t sub_garph_id) {
  int offset = C_BlockVer_offset[sub_garph_id];
  int size = C_BlockVer_num[sub_garph_id];
  std::vector<int32_t> sub_graph_st2ed(st2ed.begin() + offset,
                                        st2ed.begin() + offset + size);
  return sub_graph_st2ed;
}

int32_t fapGraph::getCurrentSubGraphId() {
  return this->current_subgraph_id;
}

std::vector<float> fapGraph::getSubGraphDistance(int32_t sub_garph_id) {
  assert(this->current_subgraph_id == sub_garph_id);
  return this->subgraph_dist;
}

std::vector<int32_t> fapGraph::getSubGraphPath(int32_t sub_garph_id) {
  assert(this->current_subgraph_id == sub_garph_id);
  return this->subgraph_path;
}

std::vector<float> fapGraph::getDistanceFromOnePoint(int32_t vectex_id) {
  assert(this->graph_id[vectex_id] == this->current_subgraph_id);
  int32_t current_index = this->ed2st[vectex_id]
                                  - this->C_BlockVer_offset[vectex_id];
  std::vector<float> result(subgraph_dist.begin() + current_index,
      subgraph_dist.begin() + current_index + this->num_vertexs);
  return result;
}

std::vector<int32_t> fapGraph::getPathFromOnePoint(int32_t vectex_id) {
  assert(this->graph_id[vectex_id] == this->current_subgraph_id);
  int32_t current_index = this->ed2st[vectex_id]
                                  - this->C_BlockVer_offset[vectex_id];
  std::vector<int32_t> result(subgraph_path.begin() + current_index,
      subgraph_path.begin() + current_index + this->num_vertexs);
  return result;
}

float fapGraph::getDistanceP2P() {
  // TODO(Liu-xiandong):
}

std::vector<int32_t> fapGraph::getPathP2P() {
  // TODO(Liu-xiandong):
}

bool fapGraph::check_result(float *dist, int *path,
               int *source, int source_num, int vertexs,
               int *adj_size, int *row_offset,
               int *col_val, float *weight, int *graph_id) {
  fap::check_ans(dist, path, source, source_num, vertexs,
          adj_size, row_offset, col_val, weight, graph_id);
}
}  // namespace fap
