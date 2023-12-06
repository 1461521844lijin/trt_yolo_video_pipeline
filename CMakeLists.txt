cmake_minimum_required(VERSION 3.15)
project(trt_yolov8)

set(CMAKE_CXX_STANDARD 14)
# 使用cuda

# option(USE_CUDA_HW_OPTION "使用硬件编解码能力" OFF)
# if(USE_CUDA_HW_OPTION)
#     add_definitions(-DUSE_CUDA_HW)
#     message(STATUS "硬件编解码能力开启, 请确保硬件环境支持")
# endif()



set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)
set(TENSORRT_ROOT_DIR /root/trt_projects/TensorRT-8.6.1.6)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)



include_directories(/usr/local/include)
include_directories(/usr/local/include/opencv4)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(${TENSORRT_ROOT_DIR}/include)
include_directories(src)
include_directories(src/base)
include_directories(src/ffmpeg)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(${TENSORRT_ROOT_DIR}/lib)


# ffmpeg动态库
file(GLOB_RECURSE FFmpeg_LIBS 
    /usr/local/lib/libav*.so
    /usr/local/lib/libsw*.so
    /usr/local/lib/libpostproc.so)

set(
    local_libs
    pthread
    ${FFmpeg_LIBS}
    ${OpenCV_LIBS}
		nvinfer nvinfer_plugin nvonnxparser
		cuda cublas cudart cudnn
)

file(GLOB_RECURSE TRT_YOLOV8_SERVER_CPP_SRC src/*.cpp)
file(GLOB_RECURSE TRT_YOLOV8_SERVER_CU_SRC src/*.cu)

cuda_add_library(trt_yolov8_cuda_lib ${TRT_YOLOV8_SERVER_CU_SRC})
add_library(trt_yolov8_cpp_lib SHARED ${TRT_YOLOV8_SERVER_CPP_SRC})


add_executable(trt_yolov8_server test/test_infer_demo.cpp)
target_link_libraries(trt_yolov8_server trt_yolov8_cpp_lib trt_yolov8_cuda_lib ${local_libs})

add_executable(test_yolov8 test/test_example.cpp)
target_link_libraries(test_yolov8 trt_yolov8_cpp_lib trt_yolov8_cuda_lib ${local_libs})


#多路并行推拉流demo
add_executable(test_yolov8_pipeline test/test_pipeline.cpp)
target_link_libraries(test_yolov8_pipeline trt_yolov8_cpp_lib trt_yolov8_cuda_lib ${local_libs})

#多gpu负载多路并行demo
add_executable(test_yolov8_mutilgpu test/test_muitlgpu_pipeline.cpp)
target_link_libraries(test_yolov8_mutilgpu trt_yolov8_cpp_lib trt_yolov8_cuda_lib ${local_libs})