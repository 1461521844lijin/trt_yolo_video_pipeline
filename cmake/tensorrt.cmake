message("配置tensorrt环境")
set(TENSORRT_ROOT_DIR /root/trt_projects/TensorRT-8.6.1.6)

include_directories(${TENSORRT_ROOT_DIR}/include)
link_directories(${TENSORRT_ROOT_DIR}/lib)

set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser)