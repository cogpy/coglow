cmake_minimum_required(VERSION 3.1)
project(cogpy-hpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library: INTERFACE target
add_library(cog INTERFACE)
target_include_directories(cog INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Test executable
add_executable(test_all test/test_all.cpp)
target_link_libraries(test_all PRIVATE cog)

enable_testing()
add_test(NAME cogpy_tests COMMAND test_all)

# Install headers
install(DIRECTORY include/cog DESTINATION include)
