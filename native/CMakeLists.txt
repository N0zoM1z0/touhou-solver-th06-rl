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

# The Windows controller is also exercised under Wine with an embeddable
# 32-bit Python. Keep that validation DLL self-contained instead of requiring
# compiler-runtime DLLs to be copied into either the Python or retail-game
# directory.
if(MINGW)
  target_link_options(
    th06_rl_native PRIVATE -static -static-libgcc -static-libstdc++
  )
endif()
