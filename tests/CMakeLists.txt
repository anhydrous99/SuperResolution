cmake_minimum_required(VERSION 3.17)

hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)

# Remove main.cpp to not have two main() functions
list(REMOVE_AT SOURCES 0)

add_executable(tests src/tests.cpp src/test_Model.cpp src/test_ProgressBar.cpp src/test_utils.cpp src/mock_Glog.h ${SOURCES})
target_link_libraries(tests GTest::gtest_main GTest::gmock_main ${TO_LINK})
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR}/include)
add_test(NAME check COMMAND tests)
# Copy model to source directory
configure_file(${CMAKE_SOURCE_DIR}/models/G_4x.pth ${CMAKE_CURRENT_BINARY_DIR}/G_4x.pth COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/images/lr.jpg ${CMAKE_CURRENT_BINARY_DIR}/lr.jpg COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/images/sr_ESRGAN.jpg ${CMAKE_CURRENT_BINARY_DIR}/sr_ESRGAN.jpg COPYONLY)
