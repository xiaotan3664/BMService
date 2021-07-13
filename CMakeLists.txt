cmake_minimum_required(VERSION 3.5)


project(BMService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads)

aux_source_directory(./src SRC_FILES)
add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})



target_link_libraries(${CMAKE_PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
