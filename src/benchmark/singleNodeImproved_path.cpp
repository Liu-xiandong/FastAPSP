// author:lxd
// compute the APSP
// ./singleNodeImproved_path -f USpowerGrid -k 4 -direct false -weight false

#include <bits/stdc++.h>
#include "readMatFile.h"
#include "readIdFile.h"
#include "util_singleNode.h"
#include "util_Myalgorithm.h"
#include "util_centralized.h"
#include <time.h>
#include <sys/time.h>
#include "sssp.h"
#include "checkResult.h"

#define TIMER
typedef long long LL;

using std::max;
using std::stoi;
using std::string;
using std::vector;

using namespace std;

int main(int argc, char **argv)
{
    string file;
    int K;
    bool directed, weighted;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
            //ASSERT(i + 1 < argc);
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

    int vertexs;
    int edges;

    readVerEdges(vertexs, edges, file, directed, weighted);
    printf("%s %d %d %d", file.c_str(), vertexs, edges, K);
    //cout << vertexs << " " << edges << endl;

    int *adj_size = (int *)malloc(vertexs * sizeof(int));
    int *row_offset = (int *)malloc((vertexs + 1) * sizeof(int));
    int *col_val = (int *)malloc(edges * sizeof(int));
    float *weight = (float *)malloc(edges * sizeof(float));

    //step1
    //cout << "step1 start" << endl;
    //METIS 图划分结果
    int *graph_id = (int *)malloc(vertexs * sizeof(int));
    memset(graph_id, 0, vertexs * sizeof(int));
    std::unordered_map<int, std::vector<int>> BlockVer;
    //节点相关信息
    //BlockBoundary[K]用来存储K号子图里面中的边界点标号
    std::unordered_map<int, std::vector<int>> BlockBoundary;
    //记录该点是否为边界点
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

    readMatFile(vertexs, edges, adj_size, row_offset, col_val, weight, file, directed, weighted);
    //readIdFile(vertexs, graph_id, BlockVer, file, K, directed, weighted);
    readIdFile_METIS(graph_id, BlockVer, K, directed, vertexs, edges, adj_size, row_offset, col_val, weight);

    struct timeval begin, end;
    gettimeofday(&begin, NULL);
#ifdef TIMER
    struct timeval begin_init, end_init;
    struct timeval begin_floyd, end_floyd;
    struct timeval begin_sssp, end_sssp;
    struct timeval begin_minplus, end_minplus;
    struct timeval begin_reorder, end_reorder;
    double elapsedTime_init = 0;
    double elapsedTime_floyd = 0;
    double elapsedTime_sssp = 0;
    double elapsedTime_minplus = 0;
    double elapsedTime_reorder = 0;
#endif

    findBoundry(K, vertexs, graph_id, adj_size, row_offset, col_val, weight,
                BlockVer, BlockBoundary, isBoundry);

    sort_and_encode(K, vertexs, graph_id, isBoundry,
                    C_BlockVer_num, C_BlockVer_offset,
                    C_BlockBdy_num, C_BlockBdy_offset,
                    BlockVer, BlockBoundary, st2ed, ed2st);

    time_t t;
    srand((unsigned)time(&t));
    int verify_id = rand() % K + 1;
    //verify_id = 4;
    int max_sub_vertexs = 0;
    for (int i = 1; i <= K; i++)
    {
        max_sub_vertexs = max(max_sub_vertexs, C_BlockVer_num[i]);
    }
    // avoid from spill
    max_sub_vertexs += 5;

    // dist and path
    float *subGraph = (float *)malloc((LL)max_sub_vertexs * vertexs * sizeof(float));
    int *subGraph_path = (int *)malloc((LL)max_sub_vertexs * vertexs * sizeof(int));
    // verify the dist and path
    float *verify_res = (float *)malloc((LL)max_sub_vertexs * vertexs * sizeof(float));
    fill_n(verify_res, (LL)max_sub_vertexs * vertexs, MAXVALUE);
    int *verify_res_path = (int *)malloc((LL)max_sub_vertexs * vertexs * sizeof(int));
    fill_n(verify_res_path, (LL)max_sub_vertexs * vertexs, -1);

    for (int i = 1; i <= K; i++)
    {

        //step1 edge vertexs dijkstra
#ifdef TIMER
        gettimeofday(&begin_sssp, NULL);
#endif
        handle_boundry_path(subGraph, subGraph_path, vertexs, edges, C_BlockBdy_num[i],
                            adj_size, row_offset, col_val, weight, st2ed, C_BlockVer_offset[i]);

        // const long long bdy_num_tmp = C_BlockBdy_num[i];
        // for (long long j = 0; j < bdy_num_tmp * vertexs; j++)
        // {
        //     assert(subGraph_path[j] >= 0);
        // }

#ifdef TIMER
        gettimeofday(&end_sssp, NULL);
        elapsedTime_sssp += (end_sssp.tv_sec - begin_sssp.tv_sec) + (end_sssp.tv_usec - begin_sssp.tv_usec) / 1000000.0;
#endif
        //std::cout << "start floyd" << std::endl;
        //step2 inner floyd
        int sub_vertexs = C_BlockVer_num[i];
        float *subMat = (float *)malloc(sub_vertexs * sub_vertexs * sizeof(float));
        fill_n(subMat, sub_vertexs * sub_vertexs, MAXVALUE);
        int *subMat_path = (int *)malloc(sub_vertexs * sub_vertexs * sizeof(int));
        fill_n(subMat_path, sub_vertexs * sub_vertexs, -1);

        MysubMatBuild_path(subMat, subMat_path, subGraph, subGraph_path, graph_id, C_BlockVer_offset[i], sub_vertexs, C_BlockBdy_num[i],
                           vertexs, st2ed, ed2st, adj_size, row_offset, col_val, weight);

#ifdef TIMER
        gettimeofday(&begin_floyd, NULL);
#endif
        floyd_path(sub_vertexs, subMat, subMat_path);
#ifdef TIMER
        gettimeofday(&end_floyd, NULL);
        elapsedTime_floyd += (end_floyd.tv_sec - begin_floyd.tv_sec) + (end_floyd.tv_usec - begin_floyd.tv_usec) / 1000000.0;
#endif

        // for (long long j = 0; j < (long long)sub_vertexs * sub_vertexs; j++)
        // {
        //     if (subMat_path[j] < 0)
        //     {
        //         int row = st2ed[j / sub_vertexs];
        //         int col = j % sub_vertexs;
        //         int id = j / sub_vertexs;

        //         printf("the error %d (%d,%d) (%d,%d)\n", i, row, col, id, bdy_num_tmp);
        //         printf("%f %d %d\n", subMat[j], subMat_path[j], sub_vertexs);
        //     }
        //     assert(subMat_path[j] >= 0);
        // }

        //std::cout << "start minplus" << std::endl;
        //step3 gemm
        int inner_num = sub_vertexs - C_BlockBdy_num[i];
        float *mat1 = (float *)malloc(inner_num * C_BlockBdy_num[i] * sizeof(float));
#ifdef TIMER
        gettimeofday(&begin_minplus, NULL);
#endif
        MyMat1Build(mat1, subMat, inner_num, C_BlockBdy_num[i], sub_vertexs);
        long long offset = (long long)C_BlockBdy_num[i] * vertexs;

        // GPU global mem / sizeof(float)
        const double GPU_MAX_NUM = 4e9;
        const int bdy_num = C_BlockBdy_num[i];
        const double MEM_NUM = GPU_MAX_NUM / vertexs - bdy_num;
        int part_num = (int)ceil((double)inner_num / MEM_NUM);

        if (part_num == 1)
        {
            min_plus_path_advanced(mat1, subGraph, subGraph_path, subGraph + offset, subGraph_path + offset,
                                   inner_num, vertexs, C_BlockBdy_num[i]);
        }
        else
        {
            int block_size = inner_num / part_num;
            int last_size = inner_num - block_size * (part_num - 1);

            for (int i = 0; i < part_num; i++)
            {
                if (i == part_num - 1)
                {
                    min_plus_path_advanced(mat1 + i * block_size * bdy_num, subGraph, subGraph_path,
                                           subGraph + offset + (long long)i * block_size * vertexs,
                                           subGraph_path + offset + (long long)i * block_size * vertexs,
                                           last_size, vertexs, bdy_num);
                }
                else
                {
                    min_plus_path_advanced(mat1 + i * block_size * bdy_num, subGraph, subGraph_path,
                                           subGraph + offset + (long long)i * block_size * vertexs,
                                           subGraph_path + offset + (long long)i * block_size * vertexs,
                                           block_size, vertexs, bdy_num);
                }
            }
        }
        MysubMatDecode_path(subMat, subMat_path, subGraph, subGraph_path,
                            C_BlockVer_offset[i], sub_vertexs, sub_vertexs, vertexs, st2ed);

#ifdef TIMER
        gettimeofday(&end_minplus, NULL);
        elapsedTime_minplus += (end_minplus.tv_sec - begin_minplus.tv_sec) + (end_minplus.tv_usec - begin_minplus.tv_usec) / 1000000.0;
#endif
        free(subMat);
        free(subMat_path);
        free(mat1);

        if (i == verify_id)
        {
            memcpy(verify_res, subGraph, (LL)max_sub_vertexs * vertexs * sizeof(float));
            memcpy(verify_res_path, subGraph_path, (LL)max_sub_vertexs * vertexs * sizeof(float));
        }
    }

    free(subGraph);
    free(subGraph_path);
    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;

#ifdef TIMER
    printf(" %lf %lf %lf %lf ", elapsedTime, elapsedTime_floyd, elapsedTime_minplus, elapsedTime_sssp);
#endif

    //verify
    int check_num = C_BlockVer_num[verify_id];
    vector<int> source(check_num);
    for (int i = 0; i < check_num; i++)
    {
        int check_vertexs_index = C_BlockVer_offset[verify_id] + i;
        int check_vertexs = st2ed[check_vertexs_index];
        source[i] = check_vertexs;
    }
    bool check = check_ans(verify_res, verify_res_path, (int *)&source[0], check_num, vertexs,
                           adj_size, row_offset, col_val, weight, graph_id);

    if (check == false)
        printf("the %d subGraph is wrong !!!\n", verify_id);
    else
        printf("the %d subGraph is right\n", verify_id);
}