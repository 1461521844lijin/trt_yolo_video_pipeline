# 单路多实例测试
add_executable(test_yolo_detect test/test_example.cpp)
target_link_libraries(test_yolo_detect cpp_lib trt_cpp_lib cuda_lib ${CUDA_LIBS} ${TRT_LIBS} ${local_libs})

# 多路多实例测试
add_executable(test_yolo_detect_multi test/N-1-N_example.cpp)
target_link_libraries(test_yolo_detect_multi cpp_lib trt_cpp_lib cuda_lib ${CUDA_LIBS} ${TRT_LIBS} ${local_libs})

# 视频录制测试
add_executable(test_record_video test/record_example.cpp)
target_link_libraries(test_record_video cpp_lib trt_cpp_lib cuda_lib ${CUDA_LIBS} ${TRT_LIBS} ${local_libs})