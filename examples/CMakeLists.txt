cmake_minimum_required(VERSION 3.0)

project(rochade_examples)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

add_executable(chessboard chessboard.cpp)
target_link_libraries(chessboard ${OpenCV_LIBS})
target_include_directories(chessboard PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../include/)