#!/usr/bin/env python
# author: liuXiandong
# time: 2020.09.25

import os
import filecmp
import sys


def graph_mtx2metis():
    """
    mtx2metis
    """
    graphTypes = [
        "weight-directed", "weight-undirected", "unweight-directed",
        "unweight-undirected"
    ]
    cnt = 0
    for graphType in graphTypes:
        os.system("cd /home/liuxiandong/apsp/all/graph/" + graphType)
        path = os.listdir("/home/liuxiandong/apsp/all/graph/" + graphType)
        weighted_mat = []
        for p in path:
            if not os.path.isdir(p) and p[-3:] == "mtx":
                weighted_mat.append(p)
        for mat in weighted_mat:
            if cnt < 2:
                run = "./mtx2metis -f " + mat[:-4] + " -weight true"
            else:
                run = "./mtx2metis -f " + mat[:-4] + " -weight false"
            os.chdir('/home/liuxiandong/apsp/all/graph/' + graphType)
            os.system(run)
        cnt += 1


def graph_partition():
    graphTypes = [
        "weight-directed", "weight-undirected", "unweight-undirected",
        "unweight-directed"
    ]
    for graphType in graphTypes:
        os.system("cd /home/liuxiandong/apsp/all/graph/" + graphType)
        path = os.listdir("/home/liuxiandong/apsp/all/graph/" + graphType)
        weighted_mat = []
        for p in path:
            if not os.path.isdir(p) and p[-3:] == "txt":
                weighted_mat.append(p)

        partition_nums = [2, 4, 8, 16, 32, 64]
        for mat in weighted_mat:
            for partition_num in partition_nums:
                run = "gpmetis " + mat + " " + str(partition_num)
                os.chdir('/home/liuxiandong/apsp/all/graph/' + graphType)
                os.system(run)


def graph_run():
    op = " >>"
    graphTypes = [
        "weight-directed", "weight-undirected", "unweight-undirected",
        "unweight-directed"
    ]
    for graphType in graphTypes:
        os.system("cd /home/liuxiandong/apsp/all/graph/" + graphType)
        path = os.listdir("/home/liuxiandong/apsp/all/graph/" + graphType)
        weighted_mat = []
        for p in path:
            if not os.path.isdir(p) and p[-3:] == "mtx":
                weighted_mat.append(p)
        # if graphType=="unweight-undirected":
        #     weighted_mat=["lpl1.mtx","net4-1.mtx","onera_dual.mtx"]

        partition_nums = [2, 4, 8, 16, 32, 64]
        for mat in weighted_mat:
            for partition_num in partition_nums:
                if graphType == "weight-directed":
                    run_centralized = "./singleNodeCentralized -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct true -weight true " + op + " benchmark.log"
                    run_improved = "./singleNodeImproved -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct true -weight true " + op + " benchmark.log"
                if graphType == "weight-undirected":
                    run_centralized = "./singleNodeCentralized -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct false -weight true " + op + " benchmark.log"
                    run_improved = "./singleNodeImproved -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct true -weight true " + op + " benchmark.log"
                if graphType == "unweight-undirected":
                    run_centralized = "./singleNodeCentralized -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct false -weight false " + op + " benchmark.log"
                    run_improved = "./singleNodeImproved -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct false -weight false " + op + " benchmark.log"
                if graphType == "unweight-directed":
                    run_centralized = "./singleNodeCentralized -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct true -weight false " + op + " benchmark.log"
                    run_improved = "./singleNodeImproved -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct true -weight false " + op + " benchmark.log"
                # os.chdir('/home/liuxiandong/apsp/all')
                # print(run_centralized)
                # os.system(run_centralized)
                # print(run_decentralized)
                # os.system(run_decentralized)
                # print(run_improved)
                # os.system(run_improved)
            func_type = [3, 4]
            for func in func_type:
                if graphType == "weight-directed":
                    run_benchmark = "./singleNodeBenchmark -f " + mat[:-4] + " -type " + str(
                        func
                    ) + " -direct true -weight true " + op + " benchmark.log"
                if graphType == "weight-undirected":
                    run_benchmark = "./singleNodeBenchmark -f " + mat[:-4] + " -type " + str(
                        func
                    ) + " -direct false -weight true " + op + " benchmark.log"
                if graphType == "unweight-undirected":
                    run_benchmark = "./singleNodeBenchmark -f " + mat[:-4] + " -type " + str(
                        func
                    ) + " -direct false -weight false " + op + " benchmark.log"
                if graphType == "unweight-directed":
                    run_benchmark = "./singleNodeBenchmark -f " + mat[:-4] + " -type " + str(
                        func
                    ) + " -direct true -weight false " + op + " benchmark.log"
                print(run_benchmark)
                os.system(run_benchmark)


def benchmark_observationK():
    op = " >>"
    graphTypes = [
        "weight-directed", "weight-undirected", "unweight-undirected",
        "unweight-directed"
    ]
    for graphType in graphTypes:
        os.system("cd /home/liuxiandong/apsp/all/graph/" + graphType)
        path = os.listdir("/home/liuxiandong/apsp/all/graph/" + graphType)
        weighted_mat = []
        for p in path:
            if not os.path.isdir(p) and p[-3:] == "mtx":
                weighted_mat.append(p)
        if graphType == "unweight-undirected":
            weighted_mat = ["soc-Slashdot0811.mtx", "socfb-OR.mtx"]

        partition_nums = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        for mat in weighted_mat:
            for partition_num in partition_nums:
                if graphType == "weight-directed":
                    run_improved = "./singleNodeImproved_path -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct true -weight true " + op + " benchmark_observationK.log"
                if graphType == "weight-undirected":
                    run_improved = "./singleNodeImproved_path -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct true -weight true " + op + " benchmark_observationK.log"
                if graphType == "unweight-undirected":
                    run_improved = "./singleNodeImproved_path -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct false -weight false " + op + " benchmark_observationK.log"
                if graphType == "unweight-directed":
                    run_improved = "./singleNodeImproved_path -f " + mat[:-4] + " -k " + str(
                        partition_num
                    ) + " -direct true -weight false " + op + " benchmark_observationK.log"
                os.chdir('/home/liuxiandong/apsp/all')
                print(run_improved)
                os.system(run_improved)


def benchmark_singleNode():
    op = " >>"
    # graphTypes = [
    #     "weight-directed", "weight-undirected", "unweight-undirected",
    #     "unweight-directed"
    # ]
    graphTypes = ["unweight-undirected"]
    for graphType in graphTypes:
        os.system("cd /home/liuxiandong/apsp/fast/graph/" + graphType)
        path = os.listdir("/home/liuxiandong/apsp/fast/graph/" + graphType)
        weighted_mat = []
        for p in path:
            if not os.path.isdir(p) and p[-3:] == "mtx":
                weighted_mat.append(p)

        partition_nums = [8, 16, 32, 64]
        for mat in weighted_mat:
            for partition_num in partition_nums:
                if graphType == "weight-directed":
                    run_centralized = "./singleNodeCentralized_path -f " + mat[:-4] + " -k " + str(partition_num) + " -direct true -weight true " + op + " benchmark_singleNode.log"
                    run_improved = "./singleNodeImproved_path -f " + mat[:-4] + " -k " + str(partition_num) + " -direct true -weight true " + op + " benchmark_singleNode.log"
                if graphType == "weight-undirected":
                    run_centralized = "./singleNodeCentralized_path -f " + mat[:-4] + " -k " + str(partition_num) + " -direct false -weight true " + op + " benchmark_singleNode.log"
                    run_improved = "./singleNodeImproved_path -f " + mat[:-4] + " -k " + str(partition_num) + " -direct false -weight true " + op + " benchmark_singleNode.log"
                if graphType == "unweight-undirected":
                    run_centralized = "./singleNodeCentralized_path -f " + mat[:-4] + " -k " + str(partition_num) + " -direct false -weight false " + op + " benchmark_singleNode.log"
                    run_improved = "./singleNodeImproved_path -f " + mat[:-4] + " -k " + str(partition_num) + " -direct false -weight false " + op + " benchmark_singleNode.log"
                if graphType == "unweight-directed":
                    run_centralized = "./singleNodeCentralized_path -f " + mat[:-4] + " -k " + str(partition_num) + " -direct true -weight false " + op + " benchmark_singleNode.log"
                    run_improved = "./singleNodeImproved_path -f " + mat[:-4] + " -k " + str(partition_num) + " -direct true -weight false " + op + " benchmark_singleNode.log"
                os.chdir('/home/liuxiandong/apsp/fast/builds')
                print(run_centralized)
                os.system(run_centralized)
                print(run_improved)
                os.system(run_improved)
            # func_type = [1, 2, 4]
            # for func in func_type:
            #     if graphType == "weight-directed":
            #         run_benchmark = "./singleNodeBenchmark -f " + mat[:-4] + " -type " + str(
            #             func
            #         ) + " -direct true -weight true " + op + " benchmark_singleNode.log"
            #     if graphType == "weight-undirected":
            #         run_benchmark = "./singleNodeBenchmark -f " + mat[:-4] + " -type " + str(
            #             func
            #         ) + " -direct false -weight true " + op + " benchmark_singleNode.log"
            #     if graphType == "unweight-undirected":
            #         run_benchmark = "./singleNodeBenchmark -f " + mat[:-4] + " -type " + str(
            #             func
            #         ) + " -direct false -weight false " + op + " benchmark_singleNode.log"
            #     if graphType == "unweight-directed":
            #         run_benchmark = "./singleNodeBenchmark -f " + mat[:-4] + " -type " + str(
            #             func
            #         ) + " -direct true -weight false " + op + " benchmark_singleNode.log"
            #     # print(run_benchmark)
            #     # os.system(run_benchmark)


def benchmark_ksolver():
    op = " >>"
    graphTypes = [
        "weight-directed", "weight-undirected", "unweight-undirected",
        "unweight-directed"
    ]
    for graphType in graphTypes:
        os.system("cd /home/liuxiandong/apsp/all/graph/" + graphType)
        path = os.listdir("/home/liuxiandong/apsp/all/graph/" + graphType)
        weighted_mat = []
        for p in path:
            if not os.path.isdir(p) and p[-3:] == "mtx":
                weighted_mat.append(p)

        for mat in weighted_mat:
            if graphType == "weight-directed":
                run_k_solver = "./graph_solver -f " + mat[:
                                                          -4] + " -direct true -weight true -type 0 " + op + " benchmark_ksolver.log"
            if graphType == "weight-undirected":
                run_k_solver = "./graph_solver -f " + mat[:
                                                          -4] + " -direct false -weight true -type 0 " + op + " benchmark_ksolver.log"
            if graphType == "unweight-undirected":
                run_k_solver = "./graph_solver -f " + mat[:
                                                          -4] + " -direct false -weight false -type 0 " + op + " benchmark_ksolver.log"
            if graphType == "unweight-directed":
                run_k_solver = "./graph_solver -f " + mat[:
                                                          -4] + " -direct true -weight false -type 0 " + op + " benchmark_ksolver.log"
            os.chdir('/home/liuxiandong/apsp/all')
            print(run_k_solver)
            os.system(run_k_solver)


if __name__ == '__main__':
    op = " >>"
    print("############################")
    print("# Benchmark Started #")
    print("############################")
    # graph_mtx2metis()
    # graph_partition()
    # graph_run()
    # benchmark_observationK()
    benchmark_singleNode()
    #benchmark_ksolver()