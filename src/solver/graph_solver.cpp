// ./graph_solver -f soc-gowalla -direct false -weight false -k 32 -type 1

#include <bits/stdc++.h>
#include "readMatFile.h"
#include "readIdFile.h"
#include "util_preprocess.h"
#include "debug.h"
#include <time.h>
#include <sys/time.h>
#include <metis.h>

using namespace std;

int memory_message(string &file, const bool directed, const bool weighted, const int K)
{
    int vertexs;
    int edges;
    readVerEdges(vertexs, edges, file, directed, weighted);
    cout << vertexs << " " << edges << endl;
    int *adj_size = new int[vertexs];
    int *row_offset = new int[vertexs + 1];
    int *col_val = new int[edges];
    float *weight = new float[edges];

    int *graph_id = (int *)malloc(vertexs * sizeof(int));
    memset(graph_id, 0, vertexs * sizeof(int));
    unordered_map<int, vector<int>> BlockVer;
    //节点相关信息
    unordered_map<int, vector<int>> BlockBoundary; //BlockBoundary[K]用来存储K号子图里面中的边界点标号
    unordered_map<int, int> isBoundry;

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
    readIdFile_METIS(graph_id, BlockVer, K, directed, vertexs, edges, adj_size, row_offset, col_val, weight);

    findBoundry(K, vertexs, graph_id, adj_size, row_offset, col_val, weight,
                BlockVer, BlockBoundary, isBoundry);

    sort_and_encode(K, vertexs, graph_id, isBoundry,
                    C_BlockVer_num, C_BlockVer_offset,
                    C_BlockBdy_num, C_BlockBdy_offset,
                    BlockVer, BlockBoundary, st2ed, ed2st);

    const int gb = 1e9;
    for (int i = 1; i <= K; i++)
    {
        int sub_vertexs = C_BlockVer_num[i];
        int inner_num = sub_vertexs - C_BlockBdy_num[i];
        int bdy_num = C_BlockBdy_num[i];
        // the number can be stored in GPU
        const double GPU_MAX_NUM = 2e9;
        const double MEM_NUM = GPU_MAX_NUM / vertexs - bdy_num;
        printf("(%d,%d,%d)\n", bdy_num, sub_vertexs, vertexs);
        assert(MEM_NUM > 0);
        // the memory expect for the bdymat
        const double memory_no_bdymat = 1.6e10 - (double)bdy_num * vertexs * 2 * sizeof(float);
        printf("memory_no_bdymat is: %lf GB\n", memory_no_bdymat / gb);
        // the memory used to store the inner mat
        const double memory_no_inner_mat = memory_no_bdymat - (double)inner_num * bdy_num * 2 * sizeof(float);
        printf("memory_no_inner_mat is: %lf GB\n", memory_no_inner_mat / gb);
        assert(memory_no_inner_mat > (double)vertexs * 2 * sizeof(float));

        double part_size = memory_no_inner_mat / ((double)vertexs * 2 * sizeof(float));
        int part_num = (int)ceil((double)inner_num / part_size) + 1;

        printf("bdy_num is: %d inner_num is: %d ---  %lf\n", C_BlockBdy_num[i], inner_num, (double)C_BlockBdy_num[i] / sub_vertexs);
        printf("the part_num is: %d\n\n", part_num);
    }

    for (int i = 1; i <= K; i++)
    {
        double sub_num = (double)C_BlockBdy_num[i] * vertexs + (double)C_BlockVer_num[i] * C_BlockVer_num[i];
        double memory = 2 * sub_num * sizeof(float) / gb;
        printf("in the %d process,the memory is %lf GB\n", i, memory);
    }

    int max_sub_vertexs = 0;
    for (int i = 1; i <= K; i++)
    {
        int sub_vertexs = C_BlockVer_num[i];
        max_sub_vertexs = max(max_sub_vertexs, sub_vertexs);
    }
    printf("the biggest sub_vertexs is: %d\n", max_sub_vertexs);

    double alpha_ratio = 0;
    for (int i = 1; i <= K; i++)
    {
        double now_alpha = (double)C_BlockBdy_num[i] / C_BlockVer_num[i];
        alpha_ratio += now_alpha;
    }
    alpha_ratio = alpha_ratio / K;
    printf("the alpha of G is: %lf\n", alpha_ratio);
}

#define gamma ((1.0) / (640000.0))
//Solver for Optimization parameter k
// return us
double predict_time(int n, int m, int k, double alpha, double beta)
{
    //return alpha * n * beta + gamma * n * n * n / (k * k) + gamma * alpha * (1 - alpha) * n * n * n / k;
    double BandWidth = 16e9;
    double floyd_flops = 1e12;
    double minplus_flops = 2.7e12;
    double floyd_H2D = (16.0 * n * n / k) / BandWidth;
    double floyd_compute = (2.0 * n * n * n / (k * k)) / floyd_flops;
    double tmp = alpha * (1 - alpha) * n * n / k;
    double minplus_H2D = 16.0 * (tmp + (double)n * n) / BandWidth;
    double minplus_compute = 2.0 * tmp * n / minplus_flops;
    double sssp = alpha * n * beta;
    return (floyd_H2D + floyd_compute + minplus_H2D + minplus_compute) * 1000000.0 + sssp;
}

double GetAlpha(const bool directed, const int vertexs, const int edges,
                int *adj_size, int *row_offset, int *col_val, float *weight, const int k)
{
    int *id = new int[vertexs];
    std::unordered_map<int, std::vector<int>> BlockVer;
    Graph_decomposition(id, BlockVer, directed, k, vertexs, edges, adj_size, row_offset, col_val);

    std::unordered_map<int, std::vector<int>> BlockBoundary;
    std::unordered_map<int, int> isBoundry;
    findBoundry(k, vertexs, id, adj_size, row_offset, col_val, weight,
                BlockVer, BlockBoundary, isBoundry);

    int total_bdynum = 0;
    for (int i = 1; i <= k; i++)
    {
        total_bdynum += BlockBoundary[i].size();
    }

    return (double)total_bdynum / vertexs;
}

// return beta
// the time run sssp (us)
double get_sssp_time(const int vertexs, const int edges,
                     int *adj_size, int *row_offset, int *col_val, float *weight)
{
    float *res = new float[vertexs];
    int *path = new int[vertexs];
    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    dijkstra_path(0, vertexs, adj_size, row_offset, col_val, weight, res, path);
    gettimeofday(&end, NULL);
    delete[] res;
    delete[] path;
    return (end.tv_sec - begin.tv_sec) * 1000000.0 + (end.tv_usec - begin.tv_usec);
}

int solver_k(const bool directed, const int vertexs, const int edges,
             int *adj_size, int *row_offset, int *col_val, float *weight)
{
    double time = 1e12;
    int optimizationK = -1;
    const int threads = 20;
    double beta = get_sssp_time(vertexs, edges, adj_size, row_offset, col_val, weight) / threads;

    for (int k = 2; k * k <= vertexs; k *= 2)
    {
        double alpha = GetAlpha(directed, vertexs, edges, adj_size, row_offset, col_val, weight, k);
        if (predict_time(vertexs, edges, k, alpha, beta) < time)
        {
            time = predict_time(vertexs, edges, k, alpha, beta);
            optimizationK = k;
        }
    }
    // time: us
    double predict_sssp_time = beta * vertexs;

    double BandWidth = 16e9;
    double flops = 6.8e12;
    double floyd_H2D = (16.0 * vertexs * vertexs) / BandWidth;
    double floyd_compute = (2.0 * vertexs * vertexs * vertexs) / flops;
    double predict_floyd_time = (floyd_H2D + floyd_compute) * 1000000.0;

    printf("(sssp_time: %lf s)(floyd_time: %lf s)(fast_apsp_time: %lf s)\n",
           predict_sssp_time / 1000000.0, predict_floyd_time / 1000000.0, time / 1000000.0);

    if (predict_sssp_time <= predict_floyd_time && predict_sssp_time <= time)
    {
        return 0;
    }
    if (predict_floyd_time <= predict_sssp_time && predict_floyd_time <= time)
    {
        return 1;
    }
    return optimizationK;
}

void graph_message(string &file, const bool directed, const bool weighted)
{
    //CSR格式
    int vertexs;
    int edges;
    readVerEdges(vertexs, edges, file, directed, weighted);
    cout << vertexs << " " << edges << endl;
    int *adj_size = new int[vertexs];
    int *row_offset = new int[vertexs + 1];
    int *col_val = new int[edges];
    float *weight = new float[edges];
    readMatFile(vertexs, edges, adj_size, row_offset, col_val, weight, file, directed, weighted);

    int result = solver_k(directed, vertexs, edges, adj_size, row_offset, col_val, weight);

    if (result == 0)
    {
        printf("%s please use sssp\n", file.c_str());
    }
    else if (result == 1)
    {
        printf("%s please use floyd\n", file.c_str());
    }
    else
    {
        printf("%s please use fast apsp and partition is %d\n", file.c_str(), result);
    }
}

double predict_time_k(const bool directed, const int vertexs, const int edges,
                      int *adj_size, int *row_offset, int *col_val, float *weight, const int k)
{
    double time = 1e12;
    int optimizationK = -1;
    const int threads = 20;
    double beta = get_sssp_time(vertexs, edges, adj_size, row_offset, col_val, weight) / threads;

    double alpha = GetAlpha(directed, vertexs, edges, adj_size, row_offset, col_val, weight, k);
    time = predict_time(vertexs, edges, k, alpha, beta);
    return time;
}

void predict_message(string &file, const bool directed, const bool weighted, const int k)
{
    //CSR格式
    int vertexs;
    int edges;
    readVerEdges(vertexs, edges, file, directed, weighted);
    cout << vertexs << " " << edges << endl;
    int *adj_size = new int[vertexs];
    int *row_offset = new int[vertexs + 1];
    int *col_val = new int[edges];
    float *weight = new float[edges];
    readMatFile(vertexs, edges, adj_size, row_offset, col_val, weight, file, directed, weighted);

    double result = predict_time_k(directed, vertexs, edges, adj_size, row_offset, col_val, weight, k);
    printf("%s please use %d partition fast apsp time is %lf s\n", file.c_str(), k, result / 1000000.0);
}

int main(int argc, char **argv)
{
    string file;
    bool directed, weighted;
    int K;
    int type;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
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
        else if (strcmp(argv[i], "-k") == 0)
        {
            K = stoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-type") == 0)
        {
            type = stoi(argv[i + 1]);
        }
    }
    if (type == 0)
    {
        // solver the k
        graph_message(file, directed, weighted);
    }
    else if (type == 1)
    {
        // get the memory message
        memory_message(file, directed, weighted, K);
    }
    else if (type == 3)
    {
        // get the predict message
        predict_message(file, directed, weighted, K);
    }
}
