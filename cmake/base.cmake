find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(oatpp 1.3.0 REQUIRED)
find_package(oatpp-swagger 1.3.0 REQUIRED)

include_directories(/usr/local/include/opencv4)
include_directories(/usr/local/include/oatpp-1.3.0/oatpp)
include_directories(/usr/local/include/oatpp-1.3.0/oatpp-swagger)

# oatpp-swagger res path
add_definitions(-DOATPP_SWAGGER_RES_PATH="/usr/local/include/oatpp-1.3.0/bin/oatpp-swagger/res")

include_directories(${CMAKE_SOURCE_DIR}/src)


# ffmpeg动态库
file(GLOB_RECURSE FFmpeg_LIBS
        /usr/local/lib/libav*.so
        /usr/local/lib/libsw*.so
        /usr/local/lib/libpostproc.so)

link_directories(/usr/local/lib)

set(local_libs
        pthread
        ${OpenCV_LIBS}
        ${FFmpeg_LIBS}
        oatpp::oatpp
        oatpp::oatpp-swagger
        )

file(GLOB_RECURSE CPP_SRC
        ${CMAKE_SOURCE_DIR}/src/ffmpeg/*.cpp
        ${CMAKE_SOURCE_DIR}/src/graph/*.cpp
        ${CMAKE_SOURCE_DIR}/src/infer/*.cpp
        ${CMAKE_SOURCE_DIR}/src/utils/*.cpp)
add_library(cpp_lib SHARED ${CPP_SRC})


