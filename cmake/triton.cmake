message("启用triton模块")



#include(FindProtobuf)
set(triton_lib_path /workspace/install)

set(TRITON_COMMON_ENABLE_PROTOBUF ON)
set(TRITON_COMMON_ENABLE_GRPC ON)
add_compile_definitions(TRITON_ENABLE_GPU)

include_directories(/workspace/install/include)
link_directories(/workspace/install/lib)


file(GLOB_RECURSE TRITON_SRC  ${CMAKE_SOURCE_DIR}/src/triton/*.cpp)
file(GLOB_RECURSE TRITON_cc  ${CMAKE_SOURCE_DIR}/src/triton/*.cc)
add_library(triton_cpp_lib SHARED ${TRITON_SRC} ${TRITON_cc}
        ../src/triton/triton_client/TritonHttpClient.cpp
        ../src/triton/triton_client/TritonHttpClient.h)


add_executable(test_triton_client test/test_triton_client.cpp)

target_link_directories(test_triton_client PUBLIC /workspace/install/lib)
target_link_libraries(test_triton_client
    PRIVATE
        triton_cpp_lib
        grpcclient
)



add_executable(test_triton_http_client test/test_triton_http_demo.cpp)
target_link_libraries(test_triton_http_client
        triton_cpp_lib
        httpclient
        grpcclient
        protobuf
        cpp_lib
        ${local_libs}

)

add_executable(test_triton_grpc_client test/test_triton_grpc_demo.cpp)
target_link_libraries(test_triton_grpc_client
        triton_cpp_lib
        protobuf
        grpcclient
        httpclient
        cpp_lib
        cuda_lib
        ${CUDA_LIBS}
        ${local_libs}

)
