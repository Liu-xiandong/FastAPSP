find_package(HIP REQUIRED)

hip_add_library(fap_hip_kernel batch_sssp.cpp floyd.cpp minplus.cpp)