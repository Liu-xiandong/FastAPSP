# 1. Single Node
```
./singleNodeExample -f luxembourg_osm -k 128 -direct false -weight false
```

# 2. MPI node
```
mpirun -np 4 ./MoreTaskMpiExample -f luxembourg_osm -k 128 -direct false -weight false
```
