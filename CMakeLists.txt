cmake_minimum_required(VERSION 3.0)

project(rochade CXX)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_library(rochade INTERFACE)
target_include_directories(
    rochade INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(rochade INTERFACE ${OpenCV_LIBS})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rochade.hpp DESTINATION include)