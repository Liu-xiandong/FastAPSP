find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-shared;--compiler-options;-fPIC)

cuda_add_library(fap_cuda_kernel batch_sssp.cu floyd.cu minplus.cu)
# add nvgraph lib
if (WITH_NVGRAPH)
    find_library(NVGRAPH_LIBRARY nvgraph)
    target_link_libraries(fap_cuda_kernel ${NVGRAPH_LIBRARY})
endif()