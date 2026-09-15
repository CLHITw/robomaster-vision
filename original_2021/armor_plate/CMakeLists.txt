cmake_minimum_required (VERSION 2.8)
project(demo CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIR_SRC)
add_executable (demo ${DIR_SRC})

target_link_libraries(demo ${OpenCV_LIBS})
