cmake_minimum_required(VERSION 3.0)
project(BMService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_BUILD_TYPE "Debug")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH x86)
endif()


if (${TARGET_ARCH} STREQUAL "x86")
    set(LIBSOPHON_PATH /opt/sophon)
    message( "${TARGET_ARCH} mode, starting......")

    # use ffmpeg
    set(FFMPEG_DIR  ${LIBSOPHON_PATH}/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})

    # use opencv
    set(OpenCV_DIR  ${LIBSOPHON_PATH}/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    # use libbmrt libbmlib
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # include_directories(${LIBSOPHON_OPENCV_INCLUDES})
    # link_directories(${LIBSOPHON_OPENCV_LIBS})

    # include_directories(${LIBSOPHON_FFMPEG_INCLUDES})
    # link_directories(${LIBSOPHON_FFMPEG_LIBS})
   set(SophonLibs dl ${OpenCV_LIBS} ${FFMPEG_LIBS} ${the_libbmlib.so} ${the_libbmrt.so} ${the_libbmcv.so})
   message(${SophonLibs})
endif()

include_directories(src/framework)
include_directories(src/tool)
include_directories(src/model)
include_directories(thirdparty/jsonxx)
include_directories(src/lib)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/framework FRAMEWORK_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/tool TOOL_FILES)
aux_source_directory(src/model MODEL_FILES)
aux_source_directory(src/main MAIN_FILES)
set(JSONXX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/jsonxx/jsonxx.cc")


FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

#add_executable(${CMAKE_PROJECT_NAME} ${MAIN_FILES} ${TOOL_FILES} ${FRAMEWORK_FILES})
#target_link_libraries(${CMAKE_PROJECT_NAME} ${SophonLibs})

foreach(MODEL_SOURCE ${MODEL_FILES})
    get_filename_component(MODEL_NAME ${MODEL_SOURCE} NAME_WE)
    string(CONCAT MODEL_TARGET  "tpu-" ${MODEL_NAME})
    add_executable(${MODEL_TARGET} ${MODEL_SOURCE} ${TOOL_FILES} ${FRAMEWORK_FILES} ${JSONXX_SRC})
    target_link_libraries(${MODEL_TARGET} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${MODEL_TARGET} ${SophonLibs})
endforeach(MODEL_SOURCE)


aux_source_directory(src/lib LIB_SOURCES)
set(LIB_TARGET bmservice)
add_library(${LIB_TARGET} SHARED ${LIB_SOURCES} ${TOOL_FILES} ${FRAMEWORK_FILES} ${JSONXX_SRC})
target_link_libraries(${LIB_TARGET} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${LIB_TARGET} ${SophonLibs})

enable_testing()
add_subdirectory(tests)

INSTALL(TARGETS ${LIB_TARGET} LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python/bmservice/lib)
