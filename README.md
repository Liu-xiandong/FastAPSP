# Fast APSP
The Fast APSP algorithm is used to solve the All-Pairs Shortest Paths (APSP) problem. The algorithm uses the divide and conquers strategy. First, divide the graph structure by METIS, and divide the input graph G into multiple subgraphs. Then the solution of the APSP problem is solved by computing the subgraph. The Fast APSP algorithm combines the SSSP algorithm and the Floyd-Warshall algorithm. Compared with the Part APSP algorithm, it eliminates the data dependence and communication between sub-graphs. The Fast APSP algorithm has achieved good performance in graphs with good properties.

We tested a lot of sparse graph data in the Suite sparse matrix collection and network repository, and the Fast APSP algorithm showed better performance than other APSP algorithms.

## Dependency

 - g++ 7.50
 - METIS 5.1.0
 - cuda 10.1
 - Openmpi 4.0.2
 - hipcc 2.8

## Quickstart

```shell
make clean
make all
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

## Example

Here is a sample codes of APSP for graph luxembourg_osm. 

run with `./builds/singleNodeImproved_path -f luxembourg_osm -k 32 -direct false -weight false`:

## License
All the libraryies, examples, and source codes of Fast APSP are released under [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0).
