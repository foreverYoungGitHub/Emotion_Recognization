cmake_minimum_required(VERSION 3.6)
project(demo)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp emojis.cpp emojis.h)
add_executable(demo ${SOURCE_FILES})


target_link_libraries(demo ${OpenCV_LIBS} )
target_link_libraries(demo ${Caffe_LIBRARIES})