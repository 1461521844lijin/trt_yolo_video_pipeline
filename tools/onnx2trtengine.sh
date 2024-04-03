#!/bin/bash

# trtexec --onnx=workspace/yolov5s.onnx \
#     --minShapes=images:1x3x640x640 \
#     --maxShapes=images:16x3x640x640 \
#     --optShapes=images:1x3x640x640 \
#     --saveEngine=workspace/yolov5s.engine


cd /


trtexec --onnx=/root/trt_projects/infer-main/workspace/yolov8n.transd.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=/root/trt_projects/infer-main/workspace/yolov8n.transd.engine


trtexec --onnx=/root/trt_projects/infer-main/workspace/yolov8n.transd.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=/root/trt_projects/infer-main/workspace/yolov8n.transd.fp16.engine \
    --fp16

trtexec --onnx=/root/trt_projects/infer-main/workspace/yolov8n-seg.b1.transd.onnx \
    --saveEngine=/root/trt_projects/infer-main/workspace/yolov8n-seg.b1.transd.engine

trtexec --onnx=yolov5s.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:10x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=yolov5s.engine \
    --fp16