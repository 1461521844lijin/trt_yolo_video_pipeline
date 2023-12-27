find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(oatpp 1.3.0 REQUIRED)

include_directories(/usr/local/include)
include_directories(/usr/local/include/opencv4)
include_directories(/usr/local/include/oatpp-1.3.0/oatpp)

include_directories(${CMAKE_SOURCE_DIR}/src)


# ffmpeg动态库
file(GLOB_RECURSE FFmpeg_LIBS
        /usr/local/lib/libav*.so
        /usr/local/lib/libsw*.so
        /usr/local/lib/libpostproc.so)


set(local_libs
        pthread
        ${FFmpeg_LIBS}
        ${OpenCV_LIBS}
        )

file(GLOB_RECURSE CPP_SRC
        ${CMAKE_SOURCE_DIR}/src/ffmpeg/*.cpp
        ${CMAKE_SOURCE_DIR}/src/graph/*.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/*.cpp
        ${CMAKE_SOURCE_DIR}/src/utils/*.cpp)
add_library(cpp_lib SHARED ${CPP_SRC})