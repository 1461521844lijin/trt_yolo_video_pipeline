# CMake generated Testfile for 
# Source directory: /home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/highgui
# Build directory: /home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/modules/highgui
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_highgui "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/bin/opencv_test_highgui" "--gtest_output=xml:opencv_test_highgui.xml")
set_tests_properties(opencv_test_highgui PROPERTIES  LABELS "Main;opencv_highgui;Accuracy" WORKING_DIRECTORY "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVUtils.cmake;1763;add_test;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/highgui/CMakeLists.txt;309;ocv_add_accuracy_tests;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/highgui/CMakeLists.txt;0;")
