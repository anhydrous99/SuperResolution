cmake_minimum_required(VERSION 3.17)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/cpp-pm/hunter/archive/v0.23.291.tar.gz"
        SHA1 "4f57243b24a02b9c7ec5ea02c09d5c77c6a97cba"
)

project(SuperResolution)

hunter_add_package(OpenCV)
hunter_add_package(cxxopts)
hunter_add_package(glog)

find_package(OpenCV REQUIRED)
find_package(cxxopts REQUIRED)
find_package(glog REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(SuperResolution src/main.cpp src/Model.cpp src/Model.h src/utils.h src/utils.cpp src/ProgressBar.h src/ProgressBar.cpp)
target_link_libraries(SuperResolution ${OpenCV_LIBS} ${TORCH_LIBRARIES} cxxopts::cxxopts glog::glog)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/models/G_4x.pth ${CMAKE_BINARY_DIR}/G_4x.pth COPYONLY)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
