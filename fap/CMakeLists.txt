
add_library(fap SHARED fap.cpp fap_more_task_mpi.cpp)
# add metis lib
target_link_libraries(fap PUBLIC metis)
# add kernel lib
if(WITH_CUDA)
    message(STATUS "FAP: use the cuda kernel")
    add_subdirectory(kernel/cuda)
    target_link_libraries(fap PUBLIC fap_cuda_kernel)
    add_definitions(-DWITH_CUDA)
endif()
if(WITH_HIP)
    message(STATUS "FAP: use the hip kernel")
    add_subdirectory(kernel/hip)
    target_link_libraries(fap PUBLIC fap_hip_kernel)
    add_definitions(-DWITH_HIP)
endif()

# add MPI lib
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(fap PUBLIC MPI::MPI_CXX)
target_include_directories(fap PUBLIC
                           "${PROJECT_SOURCE_DIR}")