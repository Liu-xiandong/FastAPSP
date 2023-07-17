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
#include "fap/fap_more_task_mpi.h"

namespace fap {

void fapGraphMoreTaskMPI::init(int32_t num_process, int32_t K) {
    this->num_process = num_process;
    this->tasks_array.resize(K);
    this->tasks_num.resize(num_process);
    this->tasks_offset.resize(num_process);
}

// each tasks is allocated to process
void fapGraphMoreTaskMPI::balanced_tasks(int *tasks_array, int *tasks_num,
                                        int *tasks_offset, int K, int P) {
    for (int i = 0; i < K; i++) {
        tasks_array[i] = i + 1;
    }
    std::random_shuffle(tasks_array, tasks_array + K);
    int num = K / P;
    int last = K - num * (P - 1);
    assert(last > 0);

    for (int i = 0; i < P; i++) {
        if (i == P - 1)
            tasks_num[i] = last;
        else
            tasks_num[i] = num;
    }
    for (int i = 0; i < P; i++) {
        tasks_offset[i] = i * num;
    }
}

void fapGraphMoreTaskMPI::loadBalance(int32_t num_process) {
    balanced_tasks(
        this->tasks_array.data(), this->tasks_num.data(),
        this->tasks_offset.data(), this->K, this->num_process);
}

void fapGraphMoreTaskMPI::Meta_MPI_Bcast(MPI_Comm comm) {
    int32_t vertexs = this->num_vertexs;
    int32_t edges = this->num_edges;
    int32_t K = this->K;
    int32_t P = this->num_process;
    MPI_Bcast(adj_size.data(), vertexs, MPI_INT, 0, comm);
    MPI_Bcast(row_offset.data(), vertexs + 1, MPI_INT, 0, comm);
    MPI_Bcast(col_val.data(), edges, MPI_INT, 0, comm);
    MPI_Bcast(weight.data(), edges, MPI_FLOAT, 0, comm);
    MPI_Bcast(st2ed.data(), vertexs, MPI_INT, 0, comm);
    MPI_Bcast(ed2st.data(), vertexs, MPI_INT, 0, comm);
    MPI_Bcast(graph_id.data(), vertexs, MPI_INT, 0, comm);
    MPI_Bcast(C_BlockVer_num.data(), K + 1, MPI_INT, 0, comm);
    MPI_Bcast(C_BlockVer_offset.data(), K + 1, MPI_INT, 0, comm);
    MPI_Bcast(C_BlockBdy_num.data(), K + 1, MPI_INT, 0, comm);
    MPI_Bcast(C_BlockBdy_offset.data(), K + 1, MPI_INT, 0, comm);
    MPI_Bcast(tasks_array.data(), K, MPI_INT, 0, comm);
    MPI_Bcast(tasks_num.data(), P, MPI_INT, 0, comm);
    MPI_Bcast(tasks_offset.data(), P, MPI_INT, 0, comm);
}

std::vector<int> fapGraphMoreTaskMPI::getCurrentTask(int32_t process_id) {
    std::vector<int> cur_task;
    int cur_tasks_num = tasks_num[process_id];
    for (int i = 0; i < cur_tasks_num; i++) {
        int cur_tasks_offset = tasks_offset[process_id] + i;
        int cur_task_id = tasks_array[cur_tasks_offset];
        cur_task.push_back(cur_task_id);
    }
    return cur_task;
}
}  // namespace fap
