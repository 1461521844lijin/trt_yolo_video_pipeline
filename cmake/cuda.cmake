message("配置cuda环境")

#
# cuda环境的修改路径
#
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.2)

find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CUDA_LIBS ${CUDA_LIBS} cuda cublas cudart cudnn)

# 把所有的cu文件都编译成一个动态库
file(GLOB_RECURSE CU_SRC
        ${CMAKE_SOURCE_DIR}/src/cuda_kernels/*.cu
        ${CMAKE_SOURCE_DIR}/src/cuda_kernels/*.cpp
        ${CMAKE_SOURCE_DIR}/src/cuda_kernels/*.cc
        )
cuda_add_library(cuda_lib SHARED ${CU_SRC})