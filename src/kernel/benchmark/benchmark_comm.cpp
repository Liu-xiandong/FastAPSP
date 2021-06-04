#include <bits/stdc++.h>
#include <mpi.h>
#include "util_dcentralized.h"
#include "readMatFile.h"
#include "readIdFile.h"

using namespace std;

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
    //start = MPI_Wtime();

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
    int *row_offset = (int *)malloc(vertexs * sizeof(int));
    int *col_val = (int *)malloc(edges * sizeof(int));
    float *weight = (float *)malloc(edges * sizeof(float));

    //METIS 图划分结果
    int K = world_size - 1;
    int *graph_id = (int *)malloc(vertexs * sizeof(int));
    memset(graph_id, 0, vertexs * sizeof(int));
    unordered_map<int, vector<int>> BlockVer;
    //节点相关信息
    unordered_map<int, vector<int>> BlockBoundary; //BlockBoundary[K]用来存储K号子图里面中的边界点标号
    unordered_map<int, int> isBoundry;             //记录该点是否为边界点
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
        cout << "start! step1" << endl;
    if (myProcess == 0)
    {
        readMatFile(vertexs, edges, adj_size, row_offset, col_val, weight, file, directed, weighted);
        readIdFile(vertexs, graph_id, BlockVer, file, K, directed, weighted);

        findBoundry(K, vertexs, graph_id, adj_size, row_offset, col_val, weight,
                    BlockVer, BlockBoundary, isBoundry);

        sort_and_encode(K, vertexs, graph_id, isBoundry,
                        C_BlockVer_num, C_BlockVer_offset,
                        C_BlockBdy_num, C_BlockBdy_offset,
                        BlockVer, BlockBoundary, st2ed, ed2st);
    }

    //MPI Bcast
    MPI_Bcast(adj_size, vertexs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_offset, vertexs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_val, edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(weight, edges, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(st2ed, vertexs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ed2st, vertexs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(C_BlockVer_num, K + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(C_BlockVer_offset, K + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(C_BlockBdy_num, K + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(C_BlockBdy_offset, K + 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int pack = 2;
    MPI_Datatype MPI_float_pack;
    MPI_Type_contiguous(pack, MPI_FLOAT, &MPI_float_pack);
    MPI_Type_commit(&MPI_float_pack);

    int sub_vertexs;
    typedef long long LL;
    LL sub_num;
    float *subGraph;
    float *subMat;
    if (myProcess != 0)
    {
        sub_vertexs = C_BlockVer_num[myProcess];
        sub_num = (LL)sub_vertexs * vertexs;

        long long subMat_num = (long long)sub_vertexs * sub_vertexs;
        long long subMat_num_padding = (subMat_num % pack == 0) ? subMat_num : (subMat_num / pack + 1) * pack;
        subMat = (float *)malloc(subMat_num_padding * sizeof(float));
        fill_n(subMat, subMat_num_padding, MAXVALUE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    int All_diagBdy_num = 0;

    int *C_diagBdy_num = (int *)malloc((K + 1) * sizeof(int));
    memset(C_diagBdy_num, 0, (K + 1) * sizeof(int));
    int *C_diagBdy_offset = (int *)malloc((K + 1) * sizeof(int));
    memset(C_diagBdy_offset, 0, (K + 1) * sizeof(int));

    diagBdyMessageBuild(C_diagBdy_num, C_diagBdy_offset, K, C_BlockBdy_num, All_diagBdy_num);
    float *All_diagBdy = (float *)malloc(All_diagBdy_num * sizeof(float));
    int subDiagMat_num = C_BlockBdy_num[myProcess] * C_BlockBdy_num[myProcess];
    float *subDiagMat = (float *)malloc(subDiagMat_num * sizeof(float));
    subDiagMatBuild(subDiagMat, subMat, C_BlockBdy_num[myProcess], sub_vertexs);

    MPI_Allgatherv(subDiagMat, subDiagMat_num, MPI_FLOAT, All_diagBdy, C_diagBdy_num, C_diagBdy_offset, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    float *All_subMat;
    int All_subMat_num = 0;

    int *C_subMat_num = (int *)malloc((K + 1) * sizeof(int));
    memset(C_subMat_num, 0, (K + 1) * sizeof(int));
    int *C_subMat_offset = (int *)malloc((K + 1) * sizeof(int));
    memset(C_subMat_offset, 0, (K + 1) * sizeof(int));

    subMatMessageBuild(C_subMat_num, C_subMat_offset, K, C_BlockVer_num, All_subMat_num, pack);
    All_subMat = (float *)malloc((long long)All_subMat_num * pack * sizeof(float));
    fill_n(All_subMat, (long long)All_subMat_num * pack, MAXVALUE);

    //TODO MPI_Gather
    MPI_Allgatherv(subMat, C_subMat_num[myProcess], MPI_float_pack, All_subMat, C_subMat_num, C_subMat_offset, MPI_float_pack, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (myProcess == 0)
    {
        printf("comm Runtime in step4 = %f s\n", end - start);
    }

    MPI_Finalize();
}