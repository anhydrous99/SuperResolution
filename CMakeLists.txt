cmake_minimum_required(VERSION 3.17)
project(SuperResolution)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(cxxopts REQUIRED)
find_package(glog REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(SuperResolution src/main.cpp src/Model.cpp src/Model.h)
target_link_libraries(SuperResolution ${OpenCV_LIBS} ${TORCH_LIBRARIES} cxxopts::cxxopts glog::glog)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
