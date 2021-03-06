cmake_minimum_required(VERSION 3.17)

option(BUILD_TESTS "Build the tests" OFF)
option(HUNTER_ENABLE "Whether to use the hunter dependency system" ON)

if (HUNTER_ENABLE)
    include("cmake/HunterGate.cmake")
    HunterGate(
            URL "https://github.com/cpp-pm/hunter/archive/v0.23.291.tar.gz"
            SHA1 "4f57243b24a02b9c7ec5ea02c09d5c77c6a97cba"
    )
endif (HUNTER_ENABLE)

project(SuperResolution)

if (HUNTER_ENABLE)
    # Use Hunter to download and build most dependencies (except Torch)
    hunter_add_package(OpenCV)
    hunter_add_package(cxxopts)
    hunter_add_package(glog)
endif (HUNTER_ENABLE)

# Find required dependencies
find_package(OpenCV REQUIRED)
find_package(cxxopts REQUIRED)
find_package(glog REQUIRED)
find_package(Torch REQUIRED)

# Set standard to C++17
set(CMAKE_CXX_STANDARD 17)
# Add libtorch flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Libraries to link to
set(TO_LINK ${OpenCV_LIBS} ${TORCH_LIBRARIES} cxxopts::cxxopts glog::glog)

# SuperResolution sources
set(SOURCES main.cpp Model.cpp utils.cpp ProgressBar.cpp Glog.cpp)

# Prepend source directory to sources
list(TRANSFORM SOURCES PREPEND "${CMAKE_SOURCE_DIR}/src/")

# Include, Link, and Build project
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(SuperResolution ${SOURCES})
target_include_directories(SuperResolution PUBLIC include)
target_link_libraries(SuperResolution ${TO_LINK})
# Copy model to source directory
configure_file(${CMAKE_SOURCE_DIR}/models/G_4x.pth ${CMAKE_BINARY_DIR}/G_4x.pth COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/models/A_4x.pth ${CMAKE_BINARY_DIR}/A_4x.pth COPYONLY)

# To suppress visibility warnings on apple systems
if (APPLE)
    set_target_properties(SuperResolution PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif (APPLE)

# Required on windows
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET SuperResolution
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:SuperResolution>)

    # Fixes cl warning D9025
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

    # Hunter's static builds of the dependencies does not produce PDB files
    set_target_properties(SuperResolution PROPERTIES LINK_FLAGS "/ignore:4099")
endif (MSVC)

# Build Tests
if (BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif (BUILD_TESTS)
