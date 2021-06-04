#ifndef CHECK_RESULT_H_
#define CHECK_RESULT_H_

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sssp.h"
#include "parameter.h"

using std::string;
using std::vector;

// check the dist and path is right
// the dist and path is the part of the matrix
// source stands for the vertexs need to check
bool check_ans(float *dist, int *path, int *source, int source_num, int vertexs,
               int *adj_size, int *row_offset, int *col_val, float *weight, int *graph_id)
{

    time_t t;
    srand((unsigned)time(&t));
    int check_cases = 20;
    while (check_cases--)
    {
        // the source vertexs needed to be test
        int ver_index = rand() % source_num;
        // the real vertex id
        int ver = source[ver_index];

        int check_col_cases = 10;
        while (check_col_cases--)
        {
            // the real vertexs id
            int des = rand() % vertexs;

            float now_dist = dist[ver_index * vertexs + des];
            int bridge = path[ver_index * vertexs + des];

            // if (now_dist >= MAXVALUE / 10 || bridge == -1)
            // {
            //     bool flag = (now_dist >= MAXVALUE / 10 && bridge == -1);
            //     if (flag == false)
            //     {
            //         printf("now_dist is: %f bridge is: %d\n", now_dist, bridge);
            //         printf("ver is: %d des is %d\n", ver, des);
            //         printf("the unreached vertexs is wrong\n");

            //         if (graph_id[ver] == graph_id[des])
            //         {
            //             printf("belong to the ghost matrix\n");
            //         }
            //         return false;
            //     }
            // }

            // if (bridge == -1)
            //     continue;

            vector<float> check_dist(vertexs);
            dijkstra(ver, vertexs, adj_size, row_offset, col_val, weight, (float *)&check_dist[0]);
            float ver2bridge = check_dist[bridge];

            // unreached vertexs
            if (now_dist > MAXVALUE / 10 && check_dist[des] > MAXVALUE / 10)
                continue;

            const float eps = 1e-5;
            if (fabs(now_dist - check_dist[des]) > eps)
            {
                printf("the dist is wrong!\n");
                printf("%f - %f \n", check_dist[des], now_dist);
                if (graph_id[ver] == graph_id[des])
                {
                    printf("wrong in the floyd\n");
                }
                else
                {
                    printf("maybe wrong in the minplus\n");
                }
                return false;
            }

            // dijkstra(bridge, vertexs, adj_size, row_offset, col_val, weight, (float *)&check_dist[0]);
            // float bridge2des = check_dist[des];
            // if (fabs(ver2bridge + bridge2des - now_dist) > eps)
            // {
            //     printf("the path is wrong!\n");
            //     return false;
            // }
        }
    }
    return true;
}

#endif