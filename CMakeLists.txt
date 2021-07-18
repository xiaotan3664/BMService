cmake_minimum_required(VERSION 3.0)


project(BMService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")

set(BMSDK_PATH $ENV{REL_TOP})
set(HOST_ARCH pcie)

include_directories(src/framework)
include_directories(src/model)
include_directories(${BMSDK_PATH}/include/bmlib)
include_directories(${BMSDK_PATH}/include/bmruntime)

link_directories(${BMSDK_PATH}/lib/bmnn/${HOST_ARCH})
link_directories(${BMSDK_PATH}/lib/bmnn/${HOST_ARCH})

find_package(Threads)

aux_source_directory(./src/framework FRAMEWORK_FILES)
aux_source_directory(./src/model MODEL_FILES)
aux_source_directory(./src/main MAIN_FILES)

add_executable(${CMAKE_PROJECT_NAME} ${MAIN_FILES} ${FRAMEWORK_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${CMAKE_PROJECT_NAME} bmrt bmlib)
foreach(MODEL_SOURCE ${MODEL_FILES})
    message(${MODEL_SOURCE})
    get_filename_component(MODEL_NAME ${MODEL_SOURCE} NAME_WE)
    string(CONCAT MODEL_TARGET  "BMService-" ${MODEL_NAME})
    message(${MODEL_TARGET})
    add_executable(${MODEL_TARGET} ${MODEL_SOURCE} ${FRAMEWORK_FILES})
    target_link_libraries(${MODEL_TARGET} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${MODEL_TARGET} bmrt bmlib)
endforeach(MODEL_SOURCE)

