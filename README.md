# Fast-APSP
The Fast APSP algorithm is used to solve the All-Pairs Shortest Paths (APSP) problem. The algorithm uses the divide and conquers strategy. First, divide the graph structure by METIS, and divide the input graph G into multiple subgraphs. Then the solution of the APSP problem is solved by computing the subgraph. The Fast APSP algorithm combines the SSSP algorithm and the Floyd-Warshall algorithm. Compared with the Part APSP algorithm, it eliminates the data dependence and communication between sub-graphs. The Fast APSP algorithm has achieved good performance in graphs with good properties. The algorithm diagram is as follows.

<p align="center"><img src=doc/fig.png></p>

We tested a lot of sparse graph data in the Suite sparse matrix collection and network repository, and the Fast APSP algorithm showed better performance than other APSP algorithms.

## Dependency

 - g++ 7.50
 - METIS 5.1.0
 - cuda 10.1
 - Openmpi 4.0.2

## Quickstart

```shell
mkdir build
cd build
cmake ..
```

## Usage

Here are the basic usages of Fast APSP.

```shell
./EXE -f <graph> [options]

[-f] Choose a graph which must in ./graph directory

[-direct] Whether the input graph is a directed graph (DeFault: false)

[-weight] Whether the input graph is a weighted graph (DeFault: false)

[-k] Number of graph divisions
```

The following interface code is performed.

```
#include "fap/fap.h"

fap::fapGraph G(file, directed, weighted, K);
G.preCondition();
if (G.isSplit()) {
    G.solveSubGraph(1, true);  // Solve the first subgraph.
} else {
    G.solve();  // Solve all subgraphs.
}
```

## Example

Here is a sample codes of APSP for graph luxembourg_osm. 

run with `./singleNodeExample -f luxembourg_osm -k 128 -direct false -weight false`:

## Cite Us

If you use our library, please cite our research paper.

```
@inproceedings{yang2023fast,
  title={Fast All-Pairs Shortest Paths Algorithm in Large Sparse Graph},
  author={Yang, Shaofeng and Liu, Xiandong and Wang, Yunting and He, Xin and Tan, Guangming},
  booktitle={Proceedings of the 37th International Conference on Supercomputing},
  pages={277--288},
  year={2023}
}
```

## License
All the libraryies, examples, and source codes of Fast APSP are released under [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).
