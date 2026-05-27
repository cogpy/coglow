cmake_minimum_required(VERSION 3.15)
project(coglow VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_TESTS "Build CogLow smoke tests" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cog INTERFACE)
target_include_directories(cog INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cog INTERFACE cxx_std_11)

add_library(coglow INTERFACE)
target_link_libraries(coglow INTERFACE cog)
add_library(coglow::coglow ALIAS coglow)

if(BUILD_TESTS)
    enable_testing()
    add_executable(coglow_smoke tests/coglow_smoke.cpp)
    target_link_libraries(coglow_smoke PRIVATE coglow)
    add_test(NAME coglow_smoke COMMAND coglow_smoke)
endif()

install(DIRECTORY include/cog DESTINATION include)
