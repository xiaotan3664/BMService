
find_package(GTest REQUIRED)
foreach(name testBMQueue testBMPipeline)
    add_executable(${name} ${name}.cpp ${FRAMEWORK_FILES} ${JSONXX_SRC} ${TOOL_FILES})
    target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE ${GTEST_BOTH_LIBRARIES} ${SophonLibs})
    add_test(${name} ${CMAKE_CURRENT_BINARY_DIR}/${name})
endforeach()

add_executable(testOpenCVLinkage testOpenCVLinkage.cpp)
target_link_libraries(testOpenCVLinkage PRIVATE ${SophonLibs})

