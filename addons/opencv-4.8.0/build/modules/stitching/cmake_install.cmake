# Install script for directory: /home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibsx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so.4.8.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so.408"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/usr/local/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY OPTIONAL FILES
    "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/lib/libopencv_stitching.so.4.8.0"
    "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/lib/libopencv_stitching.so.408"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so.4.8.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so.408"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/lib:"
           NEW_RPATH "/usr/local/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so"
         RPATH "/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/lib/libopencv_stitching.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so"
         OLD_RPATH "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/build/lib:"
         NEW_RPATH "/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_stitching.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/autocalib.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/blenders.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/camera.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/exposure_compensate.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/matchers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/motion_estimators.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/seam_finders.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/timelapsers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/util.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/util_inl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/warpers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching/detail" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/detail/warpers_inl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/stitching" TYPE FILE OPTIONAL FILES "/home/my/workflow2024/01_project/trt_yolo_video_pipeline/addons/opencv-4.8.0/modules/stitching/include/opencv2/stitching/warpers.hpp")
endif()

