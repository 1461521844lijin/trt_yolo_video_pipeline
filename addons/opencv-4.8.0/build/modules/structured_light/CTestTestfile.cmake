# CMake generated Testfile for 
# Source directory: /home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv_contrib-4.8.0/modules/structured_light
# Build directory: /home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/modules/structured_light
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_structured_light "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/bin/opencv_test_structured_light" "--gtest_output=xml:opencv_test_structured_light.xml")
set_tests_properties(opencv_test_structured_light PROPERTIES  LABELS "Extra;opencv_structured_light;Accuracy" WORKING_DIRECTORY "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVUtils.cmake;1763;add_test;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1133;ocv_add_accuracy_tests;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv_contrib-4.8.0/modules/structured_light/CMakeLists.txt;2;ocv_define_module;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv_contrib-4.8.0/modules/structured_light/CMakeLists.txt;0;")
