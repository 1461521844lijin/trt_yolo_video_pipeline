cmake_minimum_required(VERSION 3.15)
project(VideoPipeLine)

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



add_executable(trt_yolov8_server test/test_infer_demo.cpp)
target_link_libraries(trt_yolov8_server cpp_lib cuda_lib ${CUDA_LIBS} ${TRT_LIBS} ${local_libs})

add_executable(test_yolov8 test/test_example.cpp)
target_link_libraries(test_yolov8 cpp_lib cuda_lib ${CUDA_LIBS} ${TRT_LIBS} ${local_libs})


#多路并行推拉流demo
add_executable(test_yolov8_pipeline test/test_pipeline.cpp)
target_link_libraries(test_yolov8_pipeline cpp_lib cuda_lib ${CUDA_LIBS} ${TRT_LIBS} ${local_libs})

#多gpu负载多路并行demo
add_executable(test_yolov8_mutilgpu test/test_muitlgpu_pipeline.cpp)
target_link_libraries(test_yolov8_mutilgpu cpp_lib cuda_lib ${CUDA_LIBS} ${TRT_LIBS} ${local_libs})