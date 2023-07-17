cmake_minimum_required(VERSION 3.15)

# set the project name and version
project(fapApp VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(WITH_CUDA "Enable the CUDA backend for supported interfaces" ON)
option(WITH_HIP "Enable the HIP backend for supported interfaces" OFF)

# add the fap library
add_subdirectory(fap)
list(APPEND EXTRA_LIBS fap)
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/fap")

# Find all *Example.cpp files and store in list examples
file(GLOB_RECURSE examples RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*Example.cpp")
foreach(examplefile IN LISTS examples)
    # Get file name without directory
    get_filename_component(examplename ${examplefile} NAME_WE)
    add_executable(${examplename} ${examplefile})
    # add include lib
    target_link_libraries(${examplename} PUBLIC metis)
    target_link_libraries(${examplename} PUBLIC fap)
    find_package(MPI REQUIRED)
    target_link_libraries(${examplename} PUBLIC MPI::MPI_CXX)

    # add include path
    target_include_directories(${examplename} PUBLIC
                              "${PROJECT_SOURCE_DIR}"
                              "${EXTRA_INCLUDES}"
                              )
endforeach()
