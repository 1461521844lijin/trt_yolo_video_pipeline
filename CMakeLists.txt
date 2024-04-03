cmake_minimum_required(VERSION 3.15)
project(VideoPipeline)

set(CMAKE_CXX_STANDARD 17)

OPTION(ENABLE_CUDA "使用CUDA,TensorRT" ON)
OPTION(ENABLE_TENSORRT "使用TensorRT" ON)
OPTION(ENABLE_HWCODEC "使用硬件编解码能力" OFF)


# 基础代码依赖配置
include(${CMAKE_SOURCE_DIR}/cmake/base.cmake)

if (ENABLE_CUDA)
    # CUDA依赖配置
    include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake)
endif ()
if (ENABLE_TENSORRT)
    # TensorRT依赖配置
    include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake)
endif ()


cuda_add_executable(test_yolo_detect test/test_example_1.cpp)
target_link_libraries(test_yolo_detect cpp_lib trt_cpp_lib cuda_lib ${CUDA_LIBS} ${TRT_LIBS} ${local_libs} -lcurl -ljsoncpp)

# add_executable(test_yolo_detect test/test_http.cpp)
# target_link_libraries(test_yolo_detect -lcurl -ljsoncpp)

