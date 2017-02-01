cmake_minimum_required(VERSION 3.2.2)
project (readDetections)
SET(GCC_COVERAGE_COMPILE_FLAGS "-g")

set(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
#set(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_DEBUG)

link_directories(/usr/local/lib/)
include_directories(/usr/local/include)
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
add_executable(readDetections main.cpp DetectionReader.h DetectionReader.cpp)
set_property(TARGET readDetections PROPERTY CXX_STANDARD 11)
target_link_libraries(readDetections opencv_core opencv_highgui opencv_imgproc)
