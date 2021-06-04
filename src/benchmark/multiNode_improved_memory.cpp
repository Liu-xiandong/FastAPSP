// author:lxd
// compute the APSP
// tasks == processs
// memory optimize
// mpirun -np 4 ./multiNode_improved_memory -f USpowerGrid -direct false -weight false

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
    bool directed, weighted;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
            //ASSERT(i + 1 < argc);
            file = argv[i + 1];
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
    int K = world_size;
    int *graph_id = (int *)malloc(vertexs * sizeof(int));
    memset(graph_id, 0, vertexs * sizeof(int));
    std::unordered_map<int, std::vector<int>> BlockVer;
    //节点相关信息
    std::unordered_map<int, std::vector<int>> BlockBoundary; //BlockBoundary[K]用来存储K号子图里面中的边界点标号
    std::unordered_map<int, int> isBoundry;                  //记录该点是否为边界点
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

    //step1
    if (myProcess == 1)
        cout << "start! step1" << endl;
    int sub_vertexs;
    int bdy_num;
    long long sub_num;
    long long sub_num_padding;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    const int mysubGraph_id = myProcess + 1;
    const int pack = 2;

    //step1 edge vertexs dijkstra
    sub_vertexs = C_BlockVer_num[mysubGraph_id];
    bdy_num = C_BlockBdy_num[mysubGraph_id];
    sub_num = bdy_num * vertexs;
    sub_num_padding = (sub_num % pack == 0) ? sub_num : (sub_num / pack + 1) * pack;
    float *subGraph = (float *)malloc(sub_num_padding * sizeof(float));
    int *subGraph_path = (int *)malloc(sub_num_padding * sizeof(int));
    if (subGraph == NULL)
    {
        printf("ATTENTION! the memory can not be allocated!\n");
    }

    handle_boundry_path(subGraph, subGraph_path, vertexs, edges, bdy_num,
                        adj_size, row_offset, col_val, weight, st2ed, C_BlockVer_offset[mysubGraph_id]);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (myProcess == 0)
    {
        printf("sssp total Runtime = %f s\n", end - start);
    }
    //step2 inner floyd
    if (myProcess == 1)
        cout << "start! step2" << endl;

    long long subMat_len = (long long)sub_vertexs * sub_vertexs;
    float *subMat = (float *)malloc(subMat_len * sizeof(float));
    int *subMat_path = (int *)malloc(subMat_len * sizeof(float));
    fill_n(subMat, subMat_len, MAXVALUE);
    fill_n(subMat_path, subMat_len, -1);

    MysubMatBuild_path(subMat, subMat_path, subGraph, subGraph_path, graph_id, C_BlockVer_offset[mysubGraph_id],
                       sub_vertexs, bdy_num, vertexs, st2ed, ed2st,
                       adj_size, row_offset, col_val, weight);

    MPI_Barrier(MPI_COMM_WORLD);
    //step3 gemm
    if (myProcess == 1)
        cout << "start floyd!" << endl;

    floyd_path(sub_vertexs, subMat, subMat_path);

    MPI_Barrier(MPI_COMM_WORLD);
    //step3 gemm
    if (myProcess == 1)
        cout << "start! step3" << endl;

    int inner_num = sub_vertexs - bdy_num;
    float *mat1 = (float *)malloc(inner_num * bdy_num * sizeof(float));
    MyMat1Build(mat1, subMat, inner_num, bdy_num, sub_vertexs);

    // the number can be stored in GPU
    const int gb = 1e9;
    const double GPU_MAX_NUM = 2e9;
    // the memory expect for the bdymat
    const double memory_no_bdymat = 1.6e10 - (double)bdy_num * vertexs * 2 * sizeof(float);
    assert(memory_no_bdymat > 0);
    // the memory used to store the inner mat
    const double memory_no_inner_mat = memory_no_bdymat - (double)inner_num * bdy_num * 2 * sizeof(float);
    assert(memory_no_inner_mat > (double)vertexs * 2 * sizeof(float));
    double part_size = memory_no_inner_mat / ((double)vertexs * 2 * sizeof(float));
    int part_num = (int)ceil((double)inner_num / part_size);

    if (part_num == 1)
    {
        float *subGraph_tmp = new float[inner_num * vertexs];
        int *subGraph_path_tmp = new int[inner_num * vertexs];
        min_plus_path_advanced(mat1, subGraph, subGraph_path, subGraph_tmp, subGraph_path_tmp,
                               inner_num, vertexs, bdy_num);

        int subMat_offset = bdy_num * sub_vertexs;
        MysubMatDecode_path(subMat + subMat_offset, subMat_path + subMat_offset,
                            subGraph_tmp, subGraph_path_tmp, C_BlockVer_offset[mysubGraph_id],
                            sub_vertexs - bdy_num, sub_vertexs,
                            vertexs, st2ed);
        // store data to disk
        // #ifdef DEBUG
        //         // verify the ans
        //         if (true)
        //         {
        //             int check_num = C_BlockVer_num[mysubGraph_id];
        //             vector<int> source(check_num);
        //             for (int i = 0; i < check_num; i++)
        //             {
        //                 int check_vertexs_index = C_BlockVer_offset[mysubGraph_id] + i + bdy_num;
        //                 int check_vertexs = st2ed[check_vertexs_index];
        //                 source[i] = check_vertexs;
        //             }
        //             bool check = check_ans(subGraph_tmp, subGraph_path_tmp, (int *)&source[0], check_num, vertexs,
        //                                    adj_size, row_offset, col_val, weight, graph_id);

        //             if (check == false)
        //                 printf("the %d subGraph is wrong !!!\n", mysubGraph_id);
        //             else
        //                 printf("the %d subGraph is right\n", mysubGraph_id);
        //         }
        // #endif
        // delete[] subGraph_tmp;
        // delete[] subGraph_path_tmp;
    }
    else
    {
        int block_size = inner_num / part_num;
        int last_size = inner_num - block_size * (part_num - 1);
        int max_size = max(block_size, last_size);
        float *subGraph_tmp = new float[max_size * vertexs];
        int *subGraph_path_tmp = new int[max_size * vertexs];

        for (int i = 0; i < part_num; i++)
        {
            int now_row_num;
            if (i == part_num - 1)
            {
                now_row_num = last_size;
            }
            else
            {
                now_row_num = block_size;
            }

            double used_gpu_memory = 0.0;
            used_gpu_memory += (double)now_row_num * bdy_num;
            used_gpu_memory += (double)bdy_num * vertexs;
            used_gpu_memory += (double)now_row_num * vertexs;
            used_gpu_memory *= 2 * sizeof(float);
            assert(used_gpu_memory < (double)gb * 16);

            min_plus_path_advanced(mat1 + (long long)i * block_size * bdy_num, subGraph, subGraph_path,
                                   subGraph_tmp, subGraph_path_tmp,
                                   now_row_num, vertexs, bdy_num);

            long long subMat_offset = (long long)bdy_num * sub_vertexs + (long long)i * block_size * sub_vertexs;
            MysubMatDecode_path(subMat + subMat_offset, subMat_path + subMat_offset,
                                subGraph_tmp, subGraph_path_tmp, C_BlockVer_offset[mysubGraph_id],
                                now_row_num, sub_vertexs,
                                vertexs, st2ed);
            // store data to disk

            // #ifdef DEBUG
            //             //verify the ans
            //             if (i == 1)
            //             {
            //                 int check_num = now_row_num;
            //                 vector<int> source(check_num);
            //                 for (int j = 0; j < check_num; j++)
            //                 {
            //                     int check_vertexs_index = C_BlockVer_offset[mysubGraph_id] + j + i * block_size + bdy_num;
            //                     int check_vertexs = st2ed[check_vertexs_index];
            //                     source[j] = check_vertexs;
            //                 }
            //                 bool check = check_ans(subGraph_tmp, subGraph_path_tmp, (int *)&source[0], check_num, vertexs,
            //                                        adj_size, row_offset, col_val, weight, graph_id);

            //                 if (check == false)
            //                     printf("the %d subGraph is wrong !!!\n", mysubGraph_id);
            //                 else
            //                     printf("the %d subGraph is right\n", mysubGraph_id);
            //             }
            // #endif
        }
        // delete[] subGraph_tmp;
        // delete[] subGraph_path_tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (myProcess == 0)
    {
        printf("MyAlgorithm total Runtime = %f s\n", end - start);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // if (adj_size != NULL)
    //     free(adj_size);
    // if (row_offset != NULL)
    //     free(row_offset);
    // if (col_val != NULL)
    //     free(col_val);
    // if (weight != NULL)
    //     free(weight);
    // if (graph_id != NULL)
    //     free(graph_id);
    // if (st2ed != NULL)
    //     free(st2ed);
    // if (ed2st != NULL)
    //     free(ed2st);
    // if (C_BlockVer_num != NULL)
    //     free(C_BlockVer_num);
    // if (C_BlockVer_offset != NULL)
    //     free(C_BlockVer_offset);
    // if (C_BlockBdy_num != NULL)
    //     free(C_BlockBdy_num);
    // if (C_BlockBdy_offset != NULL)
    //     free(C_BlockBdy_offset);
    // if (subGraph != NULL)
    //     free(subGraph);
    // if (subMat != NULL)
    //     free(subMat);

    MPI_Finalize();
}