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

#include <time.h>
#include <sys/time.h>
#include <string.h>

#include <iostream>
#include <fstream>

#include "fap/fap.h"
#include "fap/fap_more_task_mpi.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int num_process = world_size;

    int myProcess;
    MPI_Comm_rank(MPI_COMM_WORLD, &myProcess);

    std::string file;
    int K;
    bool directed, weighted;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0) {
            file = argv[i + 1];
        } else if (strcmp(argv[i], "-k") == 0) {
            K = std::stoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-direct") == 0) {
            directed = (strcmp(argv[i + 1], "true") == 0);
        } else if (strcmp(argv[i], "-weight") == 0) {
            weighted = (strcmp(argv[i + 1], "true") == 0);
        }
    }

    // precondition
    fap::fapGraphMoreTaskMPI G(file, directed, weighted, K);
    G.init(num_process, K);
    if (myProcess == 0) {
        G.preCondition();
        G.loadBalance(num_process);
    }

    // Broadcast the meta data.
    G.Meta_MPI_Bcast(MPI_COMM_WORLD);

    // run the kernel
    auto current_task_array = G.getCurrentTask(myProcess);
    for (int i = 0; i < current_task_array.size(); i++) {
        int32_t current_task_id = current_task_array[i];
        G.solveSubGraph(current_task_id, true);
    }

    // get the result of last task
    auto current_subgraph_id = G.getCurrentSubGraphId();
    auto subgraph_dist = G.getSubGraphDistance(current_subgraph_id);
    auto subgraph_path = G.getSubGraphPath(current_subgraph_id);
    auto source = G.getSubGraphIndex(current_subgraph_id);

    // verify the last task
    bool check = G.check_result(
        subgraph_dist.data(), subgraph_path.data(),
        source.data(), source.size(), G.adj_size.size(),
        G.adj_size.data(), G.row_offset.data(),
        G.col_val.data(), G.weight.data(),
        G.getGraphId().data());

    int32_t verify_id = current_subgraph_id;
    if (check == false)
        printf("the %d subGraph is wrong !!!\n", verify_id);
    else
        printf("the %d subGraph is right\n", verify_id);

    MPI_Finalize();
}