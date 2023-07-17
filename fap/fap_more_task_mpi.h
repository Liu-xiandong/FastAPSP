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

#include <mpi.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>

namespace fap {

class fapGraphMoreTaskMPI: public fapGraph{
 private:
    void balanced_tasks(int *tasks_array, int *tasks_num,
        int *tasks_offset, int K, int P);
 public:
    // distribute the tasks to process
    std::unordered_map<int, std::vector<int>> task_per_process;
    std::vector<int32_t> tasks_array;
    std::vector<int32_t> tasks_num;
    std::vector<int32_t> tasks_offset;
    int32_t num_process;

    // contruct data struct
    // fapGraphMoreTaskMPI(std::string input_graph,
    //     bool directed, bool weight, int32_t K, int32_t num_process);

    // contruct data struct
    using fapGraph::fapGraph;
    // init the related data.
    void init(int32_t num_process, int32_t K);
    // allocate the task to different process.
    void loadBalance(int32_t num_process);
    // return meta data so that MPI can read it.
    void Meta_MPI_Bcast(MPI_Comm comm);
    // return the current task in current process.
    std::vector<int> getCurrentTask(int32_t process_id);
};
}  // namespace fap
