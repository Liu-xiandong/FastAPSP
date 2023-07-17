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

#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>

namespace fap {

class fapGraph{
 protected:
    // Graph meta and related func.
    // input param.
    std::string input_graph;
    std::stringstream input_file;

    // output param.
    // if num_vertexs is more than max_vertexs_num, split it.
    // the limit is 20k point.
    const int32_t max_vertexs_num_limit = 20000;
    bool is_split;
    bool is_path_needed;
    std::string output_file;
    std::vector<float> dist;
    std::vector<int32_t> path;
    int32_t current_subgraph_id;
    std::vector<float> subgraph_dist;
    std::vector<int32_t> subgraph_path;

    // meta graph and graph after metis.
    int32_t K;
    // to find the subgraph id which current vertex belong to.
    std::vector<int> graph_id;
    // a subgraph have vertexs which is internel points.
    std::unordered_map<int, std::vector<int>> BlockVer;
    // a subgraph have vertexs which is boundary points.
    std::unordered_map<int, std::vector<int>> BlockBoundary;
    // if a vertex is boundry or not.
    std::unordered_map<int, int> isBoundry;
    // mapping start index to end index.
    std::vector<int> st2ed;
    // mapping end index to start index.
    std::vector<int> ed2st;
    // from K subgraph, the num and offset.
    std::vector<int> C_BlockVer_num;
    std::vector<int> C_BlockVer_offset;
    std::vector<int> C_BlockBdy_num;
    std::vector<int> C_BlockBdy_offset;
    // max subgraph
    int32_t max_subgraph_vertexs;

 private:
    void run(float *subgraph_dist, int *subgraph_path,
            const int32_t subgraph_id);

 public:
    int32_t num_vertexs;
    int32_t num_edges;
    std::vector<int32_t> adj_size;
    std::vector<int32_t> row_offset;
    std::vector<int32_t> col_val;
    std::vector<float> weight;
    bool directed;
    bool weighted;

    // contruct data struct
    fapGraph(std::string input_graph,
            bool directed, bool weight, int32_t K);
    fapGraph(int32_t num_vertexs, int64_t num_edges,
            std::vector<int32_t> row_offset, std::vector<int32_t> col_val,
            std::vector<float> weight, int32_t K);

    // split graph and get the related data struct of subgraph.
    void preCondition();
    // check the memory is enough. if not, split the dist matrix.
    bool isSplit();
    // return the graph_id
    std::vector<int> getGraphId();

    // run the fast APSP algorithm. If return -1, it means the graph is too big.
    int32_t solve(bool is_path_needed = false);
    // if return -1, some thing error.
    int32_t solveSubGraph(int32_t sub_garph_id, bool is_path_needed = false);

    // return mapping
    std::vector<int32_t> getMapping();
    // return dist.
    std::vector<float> getAllDistance();
    // return path.
    std::vector<int32_t> getAllPath();
    // return current subgraph id.
    int32_t getCurrentSubGraphId();
    // return subGraph index.
    std::vector<int32_t> getSubGraphIndex(int32_t sub_garph_id);
    // return subGraph dist.
    std::vector<float> getSubGraphDistance(int32_t sub_garph_id);
    // return subGraph path.
    std::vector<int32_t> getSubGraphPath(int32_t sub_garph_id);
    // return SSSP Distance result.
    std::vector<float> getDistanceFromOnePoint(int32_t vectex_id);
    // return SSSP Path result.
    std::vector<int32_t> getPathFromOnePoint(int32_t vectex_id);
    // return distance of Point2Point
    float getDistanceP2P();
    // return path of Point2Point
    std::vector<int32_t> getPathP2P();

    // verify the result
    bool check_result(float *dist, int *path,
               int *source, int source_num, int vertexs,
               int *adj_size, int *row_offset,
               int *col_val, float *weight, int *graph_id);
};
}  // namespace fap
