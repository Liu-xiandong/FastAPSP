// author:lxd
// compute the APSP
// tasks > processs
// memory optimize
// mpirun -np 4 ./multiNode_improved_moreTask -f USpowerGrid -k 8 -direct false -weight false

#include <bits/stdc++.h>
#include <mpi.h>
#include "util_Myalgorithm.h"
#include "readMatFile.h"
#include "readIdFile.h"
#include "util_centralized.h"
#include "checkResult.h"

using namespace std;

using std::max;
using std::stoi;
using std::string;
using std::vector;
//#define OUTPUT
//#define DEBUG

//图节点编号从0开始，子图编号从1开始
int main(int argc, char **argv)
{
    //初始化MPI环境
    MPI_Init(&argc, &argv);

    //获取进程数
    //进程数必须是子图数+1  即K+1
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //获取进程号
    int myProcess;
    MPI_Comm_rank(MPI_COMM_WORLD, &myProcess);

    //获取进程的名字
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    MPI_Status status;

    float start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    string file;
    int K;
    bool directed, weighted;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
            file = argv[i + 1];
        }
        else if (strcmp(argv[i], "-k") == 0)
        {
            K = stoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-direct") == 0)
        {
            directed = (strcmp(argv[i + 1], "true") == 0);
        }
        else if (strcmp(argv[i], "-weight") == 0)
        {
            weighted = (strcmp(argv[i + 1], "true") == 0);
        }
    }

    //CSR格式
    int vertexs;
    int edges;
    if (myProcess == 0)
    {
        readVerEdges(vertexs, edges, file, directed, weighted);
        cout << vertexs << " " << edges << endl;
    }
    MPI_Bcast(&vertexs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *adj_size = (int *)malloc(vertexs * sizeof(int));
    int *row_offset = (int *)malloc((vertexs + 1) * sizeof(int));
    int *col_val = (int *)malloc(edges * sizeof(int));
    float *weight = (float *)malloc(edges * sizeof(float));

    //METIS 图划分结果
    int P = world_size;
    int *graph_id = (int *)malloc(vertexs * sizeof(int));
    memset(graph_id, 0, vertexs * sizeof(int));
    std::unordered_map<int, std::vector<int>> BlockVer;
    //节点相关信息
    //BlockBoundary[K]用来存储K号子图里面中的边界点标号
    std::unordered_map<int, std::vector<int>> BlockBoundary;
    std::unordered_map<int, int> isBoundry;
    // MPI message
    int *st2ed = (int *)malloc(vertexs * sizeof(int));
    memset(st2ed, 0, vertexs * sizeof(int));
    int *ed2st = (int *)malloc(vertexs * sizeof(int));
    memset(ed2st, 0, vertexs * sizeof(int));
    int *C_BlockVer_num = (int *)malloc((K + 1) * sizeof(int));
    memset(C_BlockVer_num, 0, (K + 1) * sizeof(int));
    int *C_BlockVer_offset = (int *)malloc((K + 1) * sizeof(int));
    memset(C_BlockVer_offset, 0, (K + 1) * sizeof(int));
    int *C_BlockBdy_num = (int *)malloc((K + 1) * sizeof(int));
    memset(C_BlockBdy_num, 0, (K + 1) * sizeof(int));
    int *C_BlockBdy_offset = (int *)malloc((K + 1) * sizeof(int));
    memset(C_BlockBdy_offset, 0, (K + 1) * sizeof(int));

    // distribute the tasks to process
    std::unordered_map<int, std::vector<int>> process_tasks;
    int *tasks_array = new int[K];
    int *tasks_num = new int[P];
    int *tasks_offset = new int[P];

    // step0
    // 读取数据
    if (myProcess == 0)
        cout << "start! step0" << endl;
    if (myProcess == 0)
    {
        readMatFile(vertexs, edges, adj_size, row_offset, col_val, weight, file, directed, weighted);
        //readIdFile(vertexs, graph_id, BlockVer, file, K, directed, weighted);
        readIdFile_METIS(graph_id, BlockVer, K, directed, vertexs, edges, adj_size, row_offset, col_val, weight);

        findBoundry(K, vertexs, graph_id, adj_size, row_offset, col_val, weight,
                    BlockVer, BlockBoundary, isBoundry);

        sort_and_encode(K, vertexs, graph_id, isBoundry,
                        C_BlockVer_num, C_BlockVer_offset,
                        C_BlockBdy_num, C_BlockBdy_offset,
                        BlockVer, BlockBoundary, st2ed, ed2st);

        balanced_tasks(C_BlockBdy_num, process_tasks, tasks_array, tasks_num, tasks_offset, K, P);
    }

    //MPI Bcast
    MPI_Bcast(adj_size, vertexs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_offset, vertexs + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_val, edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weight, edges, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(st2ed, vertexs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ed2st, vertexs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph_id, vertexs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(C_BlockVer_num, K + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(C_BlockVer_offset, K + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(C_BlockBdy_num, K + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(C_BlockBdy_offset, K + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tasks_array, K, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tasks_num, P + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tasks_offset, P + 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int pack = 2;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    int now_tasks_num = tasks_num[myProcess];
    for (int i = 0; i < now_tasks_num; i++)
    {
        int now_tasks_offset = tasks_offset[myProcess] + i;
        int now_task = tasks_array[now_tasks_offset];

        cout << "now_task " << now_task << endl;

        //compute the current task

        // parameters
        int sub_vertexs;
        int bdy_num;
        long long sub_num;
        long long sub_num_padding;
        // dist and path
        float *subGraph;
        int *subGraph_path;
        float *subMat;
        int *subMat_path;

        //step1 edge vertexs dijkstra
        sub_vertexs = C_BlockVer_num[now_task];
        bdy_num = C_BlockBdy_num[now_task];
        sub_num = (long long)bdy_num * vertexs;
        sub_num_padding = sub_num;
        subGraph = (float *)malloc(sub_num_padding * sizeof(float));
        assert(subGraph != nullptr);
        subGraph_path = (int *)malloc(sub_num_padding * sizeof(int));
        assert(subGraph_path != nullptr);

        handle_boundry_path(subGraph, subGraph_path, vertexs, edges, bdy_num,
                            adj_size, row_offset, col_val, weight, st2ed, C_BlockVer_offset[now_task]);

        //step2 inner floyd
        int subMat_len = sub_vertexs * sub_vertexs;
        subMat = (float *)malloc(subMat_len * sizeof(float));
        assert(subMat != nullptr);
        subMat_path = (int *)malloc(subMat_len * sizeof(float));
        assert(subMat_path != nullptr);

        fill_n(subMat, subMat_len, MAXVALUE);
        fill_n(subMat_path, subMat_len, -1);

        MysubMatBuild_path(subMat, subMat_path, subGraph, subGraph_path, graph_id, C_BlockVer_offset[now_task],
                           sub_vertexs, bdy_num, vertexs, st2ed, ed2st,
                           adj_size, row_offset, col_val, weight);

        floyd_path(sub_vertexs, subMat, subMat_path);

        //step3 gemm
        int inner_num = sub_vertexs - bdy_num;
        float *mat1 = new float[inner_num * bdy_num];
        assert(mat1 != nullptr);
        // MyMat1Build(mat1, subMat, inner_num, bdy_num, sub_vertexs);

        // the number can be stored in GPU
        const int gb = 1e9;
        const double GPU_MAX_NUM = 2e9;
        // the memory expect for the bdymat
        const double memory_no_bdymat = 1.6e10 - (double)bdy_num * vertexs * 2 * sizeof(float);
        assert(memory_no_bdymat > 0);

        if (memory_no_bdymat > 0)
        {
            // the memory used to store the inner mat
            const double memory_no_inner_mat = memory_no_bdymat - (double)inner_num * bdy_num * 2 * sizeof(float);
            assert(memory_no_inner_mat > (double)vertexs * 2 * sizeof(float));
            double part_size = memory_no_inner_mat / ((double)vertexs * 2 * sizeof(float));
            assert(part_size > 0);
            int part_num = 2 * (int)ceil((double)inner_num / part_size) + 1;

            if (part_num == 1)
            {
                float *subGraph_tmp = new float[(long long)inner_num * vertexs];
                assert(subGraph_tmp != nullptr);
                int *subGraph_path_tmp = new int[(long long)inner_num * vertexs];
                assert(subGraph_path_tmp != nullptr);
                min_plus_path_advanced(mat1, subGraph, subGraph_path, subGraph_tmp, subGraph_path_tmp,
                                       inner_num, vertexs, bdy_num);

                long long subMat_offset = (long long)bdy_num * sub_vertexs;
                MysubMatDecode_path(subMat + subMat_offset, subMat_path + subMat_offset,
                                    subGraph_tmp, subGraph_path_tmp, C_BlockVer_offset[now_task],
                                    sub_vertexs - bdy_num, sub_vertexs,
                                    vertexs, st2ed);
                // store data to disk
                MPI_Barrier(MPI_COMM_WORLD);
                delete[] subGraph_tmp;
                delete[] subGraph_path_tmp;
            }
            else
            {
                long long block_size = (long long)ceil((double)inner_num / part_num);
                long long last_size = inner_num - block_size * (part_num - 1);
                long long max_size = max(block_size, last_size);

                float *subGraph_tmp = new float[max_size * vertexs];
                assert(subGraph_tmp != nullptr);
                int *subGraph_path_tmp = new int[max_size * vertexs];
                assert(subGraph_path_tmp != nullptr);

                int end_index = inner_num / block_size;

                for (int j = 0; j < part_num; j++)
                {
                    int now_row_num;
                    if (j < end_index)
                    {
                        now_row_num = block_size;
                    }
                    else if (j == end_index)
                    {
                        now_row_num = inner_num % block_size;
                    }
                    else
                    {
                        now_row_num = 0;
                    }
                    if (now_row_num == 0)
                        break;

                    const double used_gpu_mem = (double)now_row_num * bdy_num + (double)bdy_num * vertexs + (double)now_row_num * vertexs;

                    // if (used_gpu_mem >= GPU_MAX_NUM)
                    // {
                    //     cout << "ATTENTION!!! used_gpu_mem is too many " << endl;
                    //     cout << "inner_num " << inner_num << " part_num " << part_num << " bdy_num " << bdy_num << endl;
                    //     cout << " now_row_num " << now_row_num << endl;
                    // }
                    // if (now_row_num < 0)
                    // {
                    //     cout << " now_row_num < 0 :" << now_row_num << endl;
                    // }

                    assert(now_row_num > 0);
                    assert(used_gpu_mem < GPU_MAX_NUM);
                    min_plus_path_advanced(mat1 + j * block_size * bdy_num, subGraph, subGraph_path,
                                           subGraph_tmp, subGraph_path_tmp,
                                           now_row_num, vertexs, bdy_num);

                    long long subMat_offset = (long long)bdy_num * sub_vertexs + (long long)j * block_size * sub_vertexs;
                    MysubMatDecode_path(subMat + subMat_offset, subMat_path + subMat_offset,
                                        subGraph_tmp, subGraph_path_tmp, C_BlockVer_offset[now_task],
                                        now_row_num, sub_vertexs,
                                        vertexs, st2ed);
                    // store data to disk
                }
                MPI_Barrier(MPI_COMM_WORLD);
                delete[] subGraph_tmp;
                delete[] subGraph_path_tmp;
            }
        }
        else
        {
            // the boundry matrix needed to by partition
            const int bdyMat_part_num = 2;
            const int bdyMat_block_size = bdy_num / bdyMat_part_num;
            const int bdyMat_last_size = bdy_num - bdyMat_block_size * (bdyMat_part_num - 1);
            //TODO
        }

        MPI_Barrier(MPI_COMM_WORLD);
        free(subGraph);
        free(subGraph_path);
        free(subMat);
        free(subMat_path);
        delete[] mat1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (myProcess == 0)
    {
        printf("MyAlgorithm total Runtime = %f s\n", end - start);
    }

    if (adj_size != NULL)
        free(adj_size);
    if (row_offset != NULL)
        free(row_offset);
    if (col_val != NULL)
        free(col_val);
    if (weight != NULL)
        free(weight);
    if (graph_id != NULL)
        free(graph_id);
    if (st2ed != NULL)
        free(st2ed);
    if (ed2st != NULL)
        free(ed2st);
    if (C_BlockVer_num != NULL)
        free(C_BlockVer_num);
    if (C_BlockVer_offset != NULL)
        free(C_BlockVer_offset);
    if (C_BlockBdy_num != NULL)
        free(C_BlockBdy_num);
    if (C_BlockBdy_offset != NULL)
        free(C_BlockBdy_offset);

    MPI_Finalize();
}