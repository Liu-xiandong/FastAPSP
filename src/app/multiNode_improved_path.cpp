// author:lxd
// compute the APSP
// tasks == processs
// mpirun -np 4 ./multiNode_improved_moreTask -f USpowerGrid -direct false -weight false

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
    float floyd_start, floyd_end;
    float floyd_time = 0;
    float minplus_start, minplus_end;
    float minplus_time = 0;
    float sssp_start, sssp_end;
    float sssp_time = 0;

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
    {
        readMatFile(vertexs, edges, adj_size, row_offset, col_val, weight, file, directed, weighted);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    if (myProcess == 0)
    {
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
    long long sub_num;
    long long sub_num_padding;

    const int mysubGraph_id = myProcess + 1;
    const int pack = 2;

    //step1 edge vertexs dijkstra
    sub_vertexs = C_BlockVer_num[mysubGraph_id];
    sub_num = (long long)sub_vertexs * vertexs;
    sub_num_padding = (sub_num % pack == 0) ? sub_num : (sub_num / pack + 1) * pack;
    float *subGraph = (float *)malloc(sub_num_padding * sizeof(float));
    int *subGraph_path = (int *)malloc(sub_num_padding * sizeof(int));
    if (subGraph == NULL)
    {
        printf("ATTENTION! the memory can not be allocated!\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    sssp_start = MPI_Wtime();
    handle_boundry_path(subGraph, subGraph_path, vertexs, edges, C_BlockBdy_num[mysubGraph_id],
                        adj_size, row_offset, col_val, weight, st2ed, C_BlockVer_offset[mysubGraph_id]);

    MPI_Barrier(MPI_COMM_WORLD);
    sssp_end = MPI_Wtime();
    sssp_time += sssp_end - sssp_start;
    if (myProcess == 0)
    {
        cout << "finish sssp" << endl;
    }

    //step2 inner floyd
    int subMat_len = sub_vertexs * sub_vertexs;
    float *subMat = (float *)malloc(subMat_len * sizeof(float));
    int *subMat_path = (int *)malloc(subMat_len * sizeof(float));
    fill_n(subMat, subMat_len, MAXVALUE);
    fill_n(subMat_path, subMat_len, -1);

    MysubMatBuild_path(subMat, subMat_path, subGraph, subGraph_path, graph_id, C_BlockVer_offset[mysubGraph_id],
                       sub_vertexs, C_BlockBdy_num[mysubGraph_id], vertexs, st2ed, ed2st,
                       adj_size, row_offset, col_val, weight);

    MPI_Barrier(MPI_COMM_WORLD);
    floyd_start = MPI_Wtime();
    if (myProcess == 0)
    {
        cout << "start floyd" << endl;
    }
    floyd_path(sub_vertexs, subMat, subMat_path);

    //MPI_Barrier(MPI_COMM_WORLD);
    floyd_end = MPI_Wtime();
    floyd_time += floyd_end - floyd_start;
    // if (myProcess == 0)
    // {
    cout << "ATTENTION! finish floyd " << myProcess << endl;
    // }

    //step3 gemm
    int inner_num = sub_vertexs - C_BlockBdy_num[mysubGraph_id];
    float *mat1 = (float *)malloc(inner_num * C_BlockBdy_num[mysubGraph_id] * sizeof(float));
    MyMat1Build(mat1, subMat, inner_num, C_BlockBdy_num[mysubGraph_id], sub_vertexs);

    long long offset = (long long)C_BlockBdy_num[mysubGraph_id] * vertexs;

    // GPU global mem / sizeof(float)
    const long long GPU_MAX_NUM = 4e9;
    const int bdy_num = C_BlockBdy_num[mysubGraph_id];
    const int MEM_NUM = GPU_MAX_NUM / vertexs - bdy_num;
    int part_num = inner_num / MEM_NUM + 2;
    part_num = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    minplus_start = MPI_Wtime();
    if (part_num == 1)
    {
        min_plus_path_advanced(mat1, subGraph, subGraph_path, subGraph + offset, subGraph_path + offset, inner_num, vertexs, C_BlockBdy_num[mysubGraph_id]);
    }
    else
    {
        int block_size = inner_num / part_num;
        int last_size = inner_num - block_size * (part_num - 1);

        for (int i = 0; i < part_num; i++)
        {
            if (i == part_num - 1)
                min_plus_path_advanced(mat1 + i * block_size * bdy_num, subGraph, subGraph_path,
                                       subGraph + offset + (long long)i * block_size * vertexs,
                                       subGraph_path + offset + (long long)i * block_size * vertexs,
                                       last_size, vertexs, bdy_num);
            else
                min_plus_path_advanced(mat1 + i * block_size * bdy_num, subGraph, subGraph_path,
                                       subGraph + offset + (long long)i * block_size * vertexs,
                                       subGraph_path + offset + (long long)i * block_size * vertexs,
                                       block_size, vertexs, bdy_num);
        }
    }
    MysubMatDecode_path(subMat, subMat_path, subGraph, subGraph_path,
                        C_BlockVer_offset[mysubGraph_id], sub_vertexs, sub_vertexs, vertexs, st2ed);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    minplus_time += end - minplus_start;
    if (myProcess == 0)
    {
        cout << "finish minplus" << endl;
    }

    if (myProcess == 0)
    {
        printf("the sssp time is %f s\n", sssp_time);
        printf("the floyd time is %f s\n", floyd_time);
        printf("the minplus time is %f s\n", minplus_time);
        printf("fast Runtime = %f s\n", end - start);
    }

#ifdef OUTPUT
    if (myProcess == 0)
    {
        Mywrite2file(ALL_subGraph, vertexs, st2ed, adj_size, row_offset, col_val, weight);
    }
#else
    if (myProcess == 0)
    {
        int check_num = C_BlockVer_num[mysubGraph_id];
        vector<int> source(check_num);
        for (int i = 0; i < check_num; i++)
        {
            int check_vertexs_index = C_BlockVer_offset[mysubGraph_id] + i;
            int check_vertexs = st2ed[check_vertexs_index];
            source[i] = check_vertexs;
        }
        bool check = check_ans(subGraph, subGraph_path, (int *)&source[0], check_num, vertexs,
                               adj_size, row_offset, col_val, weight, graph_id);

        if (check == false)
            printf("the %d subGraph is wrong !!!\n", mysubGraph_id);
        else
            printf("the %d subGraph is right\n", mysubGraph_id);
    }
#endif

    MPI_Barrier(MPI_COMM_WORLD);
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
    if (subGraph != NULL)
        free(subGraph);
    if (subMat != NULL)
        free(subMat);

    MPI_Finalize();
}