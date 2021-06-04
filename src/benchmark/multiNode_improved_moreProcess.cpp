// author:lxd
// compute the APSP
// tasks < processs
// memory optimize
// mpirun -np 4 ./multiNode_improved_moreProcess -f USpowerGrid -k 2 -direct false -weight false

#include <bits/stdc++.h>
#include <mpi.h>
#include "util_Myalgorithm.h"
#include "util_centralized.h"
#include "readMatFile.h"
#include "readIdFile.h"
#include "util_moreProcess.h"
#include "checkResult.h"

using namespace std;

using std::max;
using std::stoi;
using std::string;
using std::vector;
//#define OUTPUT
#define DEBUG

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

    // current message
    const int dist_col = P / K;
    // the process belong to dist_x subGraph
    const int dist_x = myProcess / dist_col + 1;
    // the process number in current subGraph
    const int dist_y = myProcess % dist_col;
    // the boundry number of current subGraph
    const int bdy_num = C_BlockBdy_num[dist_x];
    // the vertexs number of current subGraph
    const int sub_vertexs = C_BlockVer_num[dist_x];

    /*
    step2 message
    */
    float *bdy_mat = new float[bdy_num * vertexs];
    int *bdy_mat_path = new int[bdy_num * vertexs];
    // source number of sssp task
    unordered_map<int, unordered_map<int, int>> bdy_mat_tile_size;
    // source number array of sssp task
    unordered_map<int, unordered_map<int, vector<int>>> bdy_mat_vertexs;
    // offset in bdy_mat
    unordered_map<int, unordered_map<int, int>> bdy_mat_offset;
    // build the correspending message
    build_bdymat_message(bdy_mat_tile_size, bdy_mat_vertexs, bdy_mat_offset,
                         st2ed, C_BlockBdy_num, C_BlockBdy_offset, dist_col, K, vertexs);
    // source number of current sssp task
    const int now_sssp_num = bdy_mat_tile_size[dist_x][dist_y];
    vector<int> now_sssp_array = bdy_mat_vertexs[dist_x][dist_y];
    const int now_sssp_offset = bdy_mat_offset[dist_x][dist_y];
    // distributed bdy_mat
    float *now_bdy_mat = new float[bdy_num * vertexs];
    int *now_bdy_mat_path = new int[bdy_num * vertexs];
    // solve the sssp task
    batched_sssp_path((int *)&now_sssp_array[0], now_sssp_num, vertexs, edges,
                      adj_size, row_offset, col_val, weight,
                      now_bdy_mat, now_bdy_mat_path);
    // communication
    MPI_Comm row_comm;
    int color = myProcess / dist_col;
    MPI_Comm_split(MPI_COMM_WORLD, color, myProcess, &row_comm);

    int row_size;
    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    int *bdy_mat_num = new int[row_size];
    int *bdy_mat_suboffset = new int[row_size];
    build_sssp_comm_message(bdy_mat_num, bdy_mat_suboffset, bdy_mat_tile_size,
                            dist_col, dist_x, vertexs);

    MPI_Allgatherv(now_bdy_mat, now_sssp_num * vertexs,
                   MPI_FLOAT, bdy_mat, bdy_mat_num, bdy_mat_suboffset, MPI_FLOAT, row_comm);

    MPI_Allgatherv(now_bdy_mat_path, now_sssp_num * vertexs,
                   MPI_INT, bdy_mat_path, bdy_mat_num, bdy_mat_suboffset, MPI_INT, row_comm);

    MPI_Barrier(row_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (myProcess == 0)
    {
        printf("fast apsp Runtime in sssp = %f s\n", end - start);
    }

    if (myProcess == 0)
        cout << "start! step3" << endl;
    /*
    step3 message
    */
    // the size of subMat
    unordered_map<int, unordered_map<int, int>> floyd_tile_size;
    // offset in subMat
    unordered_map<int, unordered_map<int, int>> subMat_st2ed_offset;
    const int floyd_block_size = 256;
    // build the subMat message
    build_subMat_message(floyd_tile_size, subMat_st2ed_offset, K,
                         C_BlockVer_num, C_BlockVer_offset, dist_col, floyd_block_size);

    // the current height of subMat
    const int now_floyd_tile_size = floyd_tile_size[dist_x][0];
    const int now_floyd_size = floyd_tile_size[dist_x][dist_y];

    // distributed subMat
    float *subMat = new float[now_floyd_size * sub_vertexs];
    int *subMat_path = new int[now_floyd_size * sub_vertexs];
    assert(subMat != nullptr);
    assert(subMat_path != nullptr);

    fill_n(subMat, now_floyd_size * sub_vertexs, MAXVALUE);
    fill_n(subMat_path, now_floyd_size * sub_vertexs, -1);
    // the offset in current floyd subMat
    const int now_floyd_offset = subMat_st2ed_offset[dist_x][dist_y] - C_BlockVer_offset[dist_x];

    // build the subMat from the bdy_mat and csr-G
    build_distributed_subMat(subMat, subMat_path, now_floyd_size,
                             vertexs, sub_vertexs,
                             st2ed, ed2st, graph_id, now_floyd_offset,
                             adj_size, row_offset, col_val, weight,
                             dist_y, bdy_num, now_floyd_tile_size,
                             C_BlockVer_offset[dist_x], bdy_mat, bdy_mat_path, dist_x);
    // matrix diag of floyd
    if (myProcess == 0)
        cout << "start! floyd compute" << endl;

    float *floyd_diag = new float[floyd_block_size * floyd_block_size];
    int *floyd_diag_path = new int[floyd_block_size * floyd_block_size];
    // matrix B of floyd
    float *floyd_B = new float[floyd_block_size * sub_vertexs];
    int *floyd_B_path = new int[floyd_block_size * sub_vertexs];
    // matrix A of floyd
    float *floyd_A = new float[now_floyd_size * floyd_block_size];
    int *floyd_A_path = new int[now_floyd_size * floyd_block_size];
    assert(floyd_diag != nullptr);
    assert(floyd_diag_path != nullptr);
    assert(floyd_B != nullptr);
    assert(floyd_B_path != nullptr);
    assert(floyd_A != nullptr);
    assert(floyd_A_path != nullptr);

    //compute the floyd distributed
    int floyd_step = (int)ceil((double)sub_vertexs / floyd_block_size);
    if (myProcess == 0)
    {
        cout << "need total " << floyd_step << " step!" << endl;
    }

    for (int i = 0; i < floyd_step; i++)
    {
        // if (myProcess == 0 && (i % 1 == 0))
        // {
        //     cout << "start! " << i << " step" << endl;
        // }
        int row_now_process = (floyd_block_size * i) / now_floyd_tile_size;
        if (row_now_process >= dist_col)
            row_now_process = dist_col - 1;
        if (row_rank == row_now_process)
        {
            // cordinate in local subMat
            int diag_x = (floyd_block_size * i) % now_floyd_tile_size;
            int diag_y = floyd_block_size * i;
            // compute the diag matrix
            build_floyd_diag(floyd_diag, floyd_diag_path, floyd_block_size,
                             subMat, subMat_path, now_floyd_size, sub_vertexs, diag_x, diag_y);
            floyd_path(floyd_block_size, floyd_diag, floyd_diag_path);

            // compute the B matrix
            build_floyd_B(floyd_B, floyd_B_path, floyd_block_size, sub_vertexs,
                          subMat, subMat_path, now_floyd_size, sub_vertexs, diag_x);

            floyd_path_B(floyd_B, floyd_B_path, floyd_block_size, sub_vertexs,
                         floyd_diag, floyd_diag_path);
        }
        //cout << "Bcast " <<myProcess<<" floyd_blcok_size "<<floyd_block_size<< endl;
        MPI_Bcast(floyd_B, floyd_block_size * sub_vertexs, MPI_FLOAT, row_now_process, row_comm);
        MPI_Bcast(floyd_B_path, floyd_block_size * sub_vertexs, MPI_INT, row_now_process, row_comm);
        MPI_Bcast(floyd_diag, floyd_block_size * floyd_block_size, MPI_FLOAT, row_now_process, row_comm);
        MPI_Bcast(floyd_diag_path, floyd_block_size * floyd_block_size, MPI_INT, row_now_process, row_comm);

        // compute the A matrix
        build_floyd_A(floyd_A, floyd_A_path, now_floyd_size, floyd_block_size,
                      subMat, subMat_path, now_floyd_size, sub_vertexs, floyd_block_size * i);

        floyd_path_A(floyd_A, floyd_A_path, now_floyd_size, floyd_block_size,
                     floyd_diag, floyd_diag_path);

        // compute the rest number
        // the number can be stored in GPU
        const int gb = 1e9;
        // the memory expect for the bdymat
        const double memory_no_bdymat = 1.6e10 - (double)floyd_block_size * sub_vertexs * 2 * sizeof(float);
        assert(memory_no_bdymat > 0);
        // the memory used to store the inner mat
        const double memory_no_inner_mat = memory_no_bdymat - (double)now_floyd_size * floyd_block_size * 2 * sizeof(float);
        assert(memory_no_inner_mat > (double)sub_vertexs * 2 * sizeof(float));

        double part_size = memory_no_inner_mat / ((double)sub_vertexs * 2 * sizeof(float));
        int part_num = (int)ceil((double)now_floyd_size / part_size);

        if (part_num == 1)
        {
            floyd_min_plus(floyd_A, floyd_B, floyd_B_path, subMat, subMat_path,
                           now_floyd_size, sub_vertexs, floyd_block_size);
        }
        else
        {
            floyd_minplus_partition(floyd_A, floyd_B, floyd_B_path, subMat, subMat_path,
                                    now_floyd_size, sub_vertexs, floyd_block_size, part_num);
        }

        MPI_Barrier(row_comm);
    }

    // local2global_path
    local2global_path(subMat_path, now_floyd_size, sub_vertexs, st2ed, C_BlockBdy_offset[dist_x]);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (myProcess == 0)
    {
        printf("fast apsp Runtime in floyd = %f s\n", end - start);
    }

    if (myProcess == 0)
        cout << "start! step4" << endl;
    /*
    step 4
    */
    // the size of minplus
    const int minplus_process = bdy_num / now_floyd_tile_size;
    int now_minplus_size;
    int minplus_offset = 0;
    if (row_rank < minplus_process)
    {
        now_minplus_size = 0;
    }
    else if (row_rank == minplus_process)
    {
        now_minplus_size = now_floyd_size - bdy_num % now_floyd_tile_size;
        minplus_offset = bdy_num % now_floyd_tile_size;
    }
    else
    {
        now_minplus_size = now_floyd_size;
        minplus_offset = 0;
    }

    if (now_minplus_size > 0)
    {
        // minplus A
        float *minplus_A = new float[now_minplus_size * bdy_num];
        int *minplus_A_path = new int[now_minplus_size * bdy_num];
        assert(minplus_A != nullptr);
        assert(minplus_A_path != nullptr);

        build_minplus_A(minplus_A, minplus_A_path, now_minplus_size, bdy_num,
                        subMat, subMat_path, now_floyd_size, sub_vertexs, minplus_offset);

        // the number can be stored in GPU
        const int gb = 1e9;
        const double GPU_MAX_NUM = 2e9;
        // the memory expect for the bdymat
        const double memory_no_bdymat = 1.6e10 - (double)bdy_num * vertexs * 2 * sizeof(float);
        assert(memory_no_bdymat > 0);
        // the memory used to store the inner mat
        const double memory_no_inner_mat = memory_no_bdymat - (double)now_minplus_size * bdy_num * 2 * sizeof(float);
        assert(memory_no_inner_mat > (double)vertexs * 2 * sizeof(float));
        double part_size = memory_no_inner_mat / ((double)vertexs * 2 * sizeof(float));
        int part_num = (int)ceil((double)now_minplus_size / part_size);
        cout << part_num << endl;

        if (part_num == 1)
        {
            // minplus C
            float *minplus_C = new float[(long long)now_minplus_size * vertexs];
            int *minplus_C_path = new int[(long long)now_minplus_size * vertexs];
            assert(minplus_C != nullptr);
            assert(minplus_C_path != nullptr);

            // compute C
            min_plus_path_advanced(minplus_A, bdy_mat, bdy_mat_path, minplus_C, minplus_C_path,
                                   now_minplus_size, vertexs, bdy_num);

            // floyd2minplus
            MysubMatDecode_path(subMat + minplus_offset * sub_vertexs, subMat_path + minplus_offset * sub_vertexs,
                                minplus_C, minplus_C_path, C_BlockVer_offset[dist_x], now_minplus_size, sub_vertexs, vertexs, st2ed);
        }
        else
        {
            int block_size = now_minplus_size / part_num;
            int last_size = now_minplus_size - block_size * (part_num - 1);
            int max_size = max(block_size, last_size);
            float *minplus_C = new float[(long long)max_size * vertexs];
            int *minplus_C_path = new int[(long long)max_size * vertexs];
            assert(minplus_C != nullptr);
            assert(minplus_C_path != nullptr);

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
                min_plus_path_advanced(minplus_A + (long long)i * block_size * bdy_num, bdy_mat, bdy_mat_path,
                                       minplus_C, minplus_C_path,
                                       now_row_num, vertexs, bdy_num);

                long long subMat_offset = (long long)minplus_offset * sub_vertexs + (long long)i * block_size * sub_vertexs;
                MysubMatDecode_path(subMat + subMat_offset, subMat_path + subMat_offset,
                                    minplus_C, minplus_C_path, C_BlockVer_offset[dist_x],
                                    now_row_num, sub_vertexs,
                                    vertexs, st2ed);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (myProcess == 0)
    {
        printf("fast apsp Runtime = %f s\n", end - start);
    }

    MPI_Comm_free(&row_comm);
    MPI_Finalize();
}
