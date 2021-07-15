cmake_minimum_required(VERSION 3.0)


project(BMService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BMSDK_PATH /home/bitmain/work/release/bmnnsdk2-bm1684_vSA5)
set(HOST_ARCH pcie)

include_directories(${BMSDK_PATH}/include/bmlib)
include_directories(${BMSDK_PATH}/include/bmruntime)

link_directories(${BMSDK_PATH}/lib/bmnn/${HOST_ARCH})
link_directories(${BMSDK_PATH}/lib/bmnn/${HOST_ARCH})

find_package(Threads)

aux_source_directory(./src SRC_FILES)
add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${CMAKE_PROJECT_NAME} bmlib)