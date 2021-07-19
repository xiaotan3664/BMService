cmake_minimum_required(VERSION 3.0)


project(BMService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")

if (DEFINED $ENV{REL_TOP})
    set(BMNNSDK2_TOP $ENV{REL_TOP})
else()
    set(BMNNSDK2_TOP "/home/bitmain/work/release/bmnnsdk2-bm1684_vSA5")
endif()

set(TARGET_ARCH "x86-pcie")

include_directories(src/framework)
include_directories(src/tool)
include_directories(src/model)

aux_source_directory(./src/framework FRAMEWORK_FILES)
aux_source_directory(./src/tool TOOL_FILES)
aux_source_directory(./src/model MODEL_FILES)
aux_source_directory(./src/main MAIN_FILES)

set(bmnnsdk_inc_dirs ${BMNNSDK2_TOP}/include
        ${BMNNSDK2_TOP}/include/bmruntime
        ${BMNNSDK2_TOP}/include/bmlib
        ${BMNNSDK2_TOP}/include/third_party/boost/include
        ${BMNNSDK2_TOP}/include/opencv/opencv4
        ${BMNNSDK2_TOP}/include/ffmpeg
        CACHE INTERNAL "")

set(bmnnsdk_link_libs bmrt bmlib bmcv
        opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_dnn
        avfilter avformat avcodec avutil swresample swscale bmvideo bmjpuapi bmjpulite
        boost_system boost_filesystem protobuf gflags glog pthread
        CACHE INTERNAL "")

if (${TARGET_ARCH} STREQUAL "x86-pcie")
    set(bmnnsdk_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/pcie
            ${BMNNSDK2_TOP}/lib/opencv/x86
            ${BMNNSDK2_TOP}/lib/thirdparty/x86
            ${BMNNSDK2_TOP}/lib/opencv/x86
            ${BMNNSDK2_TOP}/lib/ffmpeg/x86
            ${BMNNSDK2_TOP}/lib/decode/x86
            CACHE INTERNAL "")


elseif (${TARGET_ARCH} STREQUAL "arm-pcie")
    set(bmnnsdk_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/arm_pcie
            ${BMNNSDK2_TOP}/lib/opencv/arm_pcie
            ${BMNNSDK2_TOP}/lib/thirdparty/soc
            ${BMNNSDK2_TOP}/lib/opencv/arm_pcie
            ${BMNNSDK2_TOP}/lib/ffmpeg/arm_pcie
            ${BMNNSDK2_TOP}/lib/decode/arm_pcie
            CACHE INTERNAL "")
else()
    set(bmnnsdk_link_dirs ${BMNNSDK2_TOP}/lib/bmnn/soc
            ${BMNNSDK2_TOP}/lib/opencv/soc
            ${BMNNSDK2_TOP}/lib/thirdparty/soc
            ${BMNNSDK2_TOP}/lib/opencv/soc
            ${BMNNSDK2_TOP}/lib/ffmpeg/soc
            ${BMNNSDK2_TOP}/lib/decode/soc
            CACHE INTERNAL "")
endif ()

include_directories(${bmnnsdk_inc_dirs})
link_directories(${bmnnsdk_link_dirs})

#add_executable(${CMAKE_PROJECT_NAME} ${MAIN_FILES} ${TOOL_FILES} ${FRAMEWORK_FILES})
#target_link_libraries(${CMAKE_PROJECT_NAME} ${bmnnsdk_link_libs})

foreach(MODEL_SOURCE ${MODEL_FILES})
    get_filename_component(MODEL_NAME ${MODEL_SOURCE} NAME_WE)
    string(CONCAT MODEL_TARGET  "BMService-" ${MODEL_NAME})
    add_executable(${MODEL_TARGET} ${MODEL_SOURCE} ${TOOL_FILES} ${FRAMEWORK_FILES})
    target_link_libraries(${MODEL_TARGET} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${MODEL_TARGET} ${bmnnsdk_link_libs})
endforeach(MODEL_SOURCE)

