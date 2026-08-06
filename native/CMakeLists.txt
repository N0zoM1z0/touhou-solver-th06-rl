cmake_minimum_required(VERSION 3.20)
project(th06_rl_native LANGUAGES CXX)

add_library(th06_rl_native SHARED src/th06_rl_native.cpp)
target_compile_features(th06_rl_native PRIVATE cxx_std_20)
target_include_directories(th06_rl_native PUBLIC include)
target_compile_definitions(th06_rl_native PRIVATE TH06_RL_NATIVE_BUILD=1)

if(MSVC)
  target_compile_options(th06_rl_native PRIVATE /W4 /permissive-)
else()
  target_compile_options(th06_rl_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

