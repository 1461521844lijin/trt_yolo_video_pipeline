# CMake generated Testfile for 
# Source directory: /home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/imgproc
# Build directory: /home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/modules/imgproc
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_imgproc "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/bin/opencv_test_imgproc" "--gtest_output=xml:opencv_test_imgproc.xml")
set_tests_properties(opencv_test_imgproc PROPERTIES  LABELS "Main;opencv_imgproc;Accuracy" WORKING_DIRECTORY "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVUtils.cmake;1763;add_test;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1133;ocv_add_accuracy_tests;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/imgproc/CMakeLists.txt;13;ocv_define_module;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/imgproc/CMakeLists.txt;0;")
add_test(opencv_perf_imgproc "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/bin/opencv_perf_imgproc" "--gtest_output=xml:opencv_perf_imgproc.xml")
set_tests_properties(opencv_perf_imgproc PROPERTIES  LABELS "Main;opencv_imgproc;Performance" WORKING_DIRECTORY "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/test-reports/performance" _BACKTRACE_TRIPLES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVUtils.cmake;1763;add_test;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1274;ocv_add_test_from_target;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1134;ocv_add_perf_tests;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/imgproc/CMakeLists.txt;13;ocv_define_module;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/imgproc/CMakeLists.txt;0;")
add_test(opencv_sanity_imgproc "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/bin/opencv_perf_imgproc" "--gtest_output=xml:opencv_perf_imgproc.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_imgproc PROPERTIES  LABELS "Main;opencv_imgproc;Sanity" WORKING_DIRECTORY "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/test-reports/sanity" _BACKTRACE_TRIPLES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVUtils.cmake;1763;add_test;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1275;ocv_add_test_from_target;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/cmake/OpenCVModule.cmake;1134;ocv_add_perf_tests;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/imgproc/CMakeLists.txt;13;ocv_define_module;/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/imgproc/CMakeLists.txt;0;")