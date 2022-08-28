CU=nvcc
HIP=hipcc
CC=g++
MPICC=mpic++

# acceleted device	CPU NVIDIA AMD
USE_NVIDIA := ON
USE_AMD := OFF

# metis library
# METIS=/public/home/ncic_ict/liuxiandong/lib/metis
METIS=/home/liuxiandong/lib/metis

# dynamic library
CU_LIBS=-L/usr/local/cuda-10.1/lib64 -lcudadevrt  -L./src/kernel/cuda -lkernel -L/usr/local/cuda-10.2/lib64 -lnvgraph
HIP_LIBS=-L./src/kernel/hip -lkernel
DEP_LIBS=-L$(METIS)/lib -lmetis -fopenmp 

LIBS :=$(DEP_LIBS)
# use NVIDIA
ifeq ($(USE_NVIDIA),ON)
	LIBS += $(CU_LIBS)
endif
# use AMD
ifeq ($(USE_AMD),ON)
	LIBS += $(HIP_LIBS)
endif

#include library
INCLUDE_DIR=-I$(METIS)/include -I./src/utils -I./src/kernel
CU_INCLUDE_DIR=-I./src/utils -I/usr/local/cuda-10.1/samples/common/inc -L/usr/local/cuda-10.2/lib64 -lnvgraph

# related file in Nvidia
CU_SOURCE :=./src/kernel/cuda
CU_FILES := $(wildcard ./src/kernel/cuda/*.cu)

# related file in AMD
HIP_SOURCE :=./src/kernel/hip
HIP_FILES := $(wildcard ./src/kernel/hip/*.cpp)

# binary program
MAIN_SOURCE=./src/app
BUILD=./build
STD=c++11
CPP_FILES := $(wildcard ./src/utils/*.cpp)
SRCS := $(wildcard $(MAIN_SOURCE)/*.cpp)
OBJECTS  := $(patsubst %.cpp, %.o, $(SRCS))
TARGET  := $(patsubst %.cpp, %, $(SRCS))

# compile code in Nvidia GPU
$(CU_SOURCE)/%.o: $(CU_SOURCE)/%.cu
	$(CU) -c -Xcompiler -fPIC $^ $(CU_INCLUDE_DIR) -o $@ 
	
$(CU_SOURCE)/libkernel.so: $(CU_SOURCE)/floyd.o $(CU_SOURCE)/minplus.o $(CU_SOURCE)/sssp.o
	$(CU) -shared $(CU_SOURCE)/floyd.o $(CU_SOURCE)/minplus.o $(CU_SOURCE)/sssp.o -o $(CU_SOURCE)/libkernel.so

# compile code in AMD GPU
$(HIP_SOURCE)/%.o: $(HIP_SOURCE)/%.cpp
	$(HIP) -c -fPIC $^ $(CU_INCLUDE_DIR) -o $@ 

$(HIP_SOURCE)/libkernel.so: $(HIP_SOURCE)/floyd.o $(HIP_SOURCE)/minplus.o
	$(HIP) -shared $(HIP_SOURCE)/floyd.o $(HIP_SOURCE)/minplus.o -o $(HIP_SOURCE)/libkernel.so

# compile code in benchmark
.cpp:
	$(MPICC) -std=$(STD) $^ $(INCLUDE_DIR) $(CPP_FILES) -o $@  $(LIBS)
	mv $@ $(BUILD)

.PHONY: clean step1 step2 step3 all

# if NVIDA $(CU_SOURCE)/libkernel.so
# if AMD $(HIP_SOURCE)/libkernel.so
step1:$(CU_SOURCE)/libkernel.so

step2: $(TARGET)

step3:
	$(CU) -ccbin $(CC) -std=$(STD) -g $(INCLUDE_DIR) -o $(MAIN_SOURCE)/singleNodeBenchmark $(MAIN_SOURCE)/tmp/singleNodeBenchmark.cpp  ./src/utils/parameter.cpp -L/usr/local/cuda-10.2/lib64 -lnvgraph

all: step1 step2 step3

clean:
	rm $(BUILD)/* $(CU_SOURCE)/*.o $(MAIN_SOURCE)/*.o $(CU_SOURCE)/*.so