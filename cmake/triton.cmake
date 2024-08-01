message("启用triton模块")


option(ENABLE_TRITON_CLIENT "使用Triton" OFF)
option(ENABLE_TRITON_SERVER "使用Triton" ON)


if (ENABLE_TRITON_CLIENT)
    message("启用triton client")

    set(triton_lib_path /workspace/install)

    set(TRITON_COMMON_ENABLE_PROTOBUF ON)
    set(TRITON_COMMON_ENABLE_GRPC ON)
    add_compile_definitions(TRITON_ENABLE_GPU)

    include_directories(/workspace/install/include)
    link_directories(/workspace/install/lib)


    file(GLOB_RECURSE TRITON_SRC  ${CMAKE_SOURCE_DIR}/src/triton/triton_client/*.cpp)
    file(GLOB_RECURSE TRITON_cc  ${CMAKE_SOURCE_DIR}/src/triton/triton_client/*.cc)
    add_library(triton_cpp_lib SHARED ${TRITON_SRC} ${TRITON_cc})


    add_executable(test_triton_client test/test_triton_client.cpp)
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

    add_executable(test_triton_yolo_pipeline test/test_triton_yolo_pipeline.cpp)
    target_link_libraries(test_triton_yolo_pipeline
            triton_cpp_lib
            protobuf
            grpcclient
            httpclient
            cpp_lib
            cuda_lib
            ${CUDA_LIBS}
            ${local_libs}
    )


endif ()

if (ENABLE_TRITON_SERVER)
    message("启用triton server")

    include_directories(/opt/tritonserver/include)
    link_directories(/opt/tritonserver/lib)

    add_compile_definitions(TRITON_ENABLE_GPU)
    # TRITON_MIN_COMPUTE_CAPABILITY  根据显卡的计算能力设置
    add_compile_definitions(TRITON_MIN_COMPUTE_CAPABILITY=8.0)

    file(GLOB_RECURSE TRITON_SRC  ${CMAKE_SOURCE_DIR}/src/triton/triton_server/*.cpp)
    file(GLOB_RECURSE TRITON_cc  ${CMAKE_SOURCE_DIR}/src/triton/triton_server/*.cc)
    add_library(triton_cpp_lib SHARED ${TRITON_SRC} ${TRITON_cc})


    add_executable(test_triton_server_demo test/test_triton_server_demo.cpp)
    target_link_libraries(test_triton_server_demo
            PRIVATE
            triton_cpp_lib
            tritonserver
            cpp_lib
            cuda_lib
            ${CUDA_LIBS}
            ${local_libs}
    )


endif ()

