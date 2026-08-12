cmake_minimum_required(VERSION 3.20)
project(th06_rl_native LANGUAGES CXX)

add_library(th06_rl_native SHARED src/th06_rl_native.cpp)
target_compile_features(th06_rl_native PRIVATE cxx_std_20)
target_include_directories(th06_rl_native PUBLIC include)
target_compile_definitions(th06_rl_native PRIVATE TH06_RL_NATIVE_BUILD=1)

add_library(th06_rl_ranker SHARED src/th06_rl_ranker.cpp)
target_compile_features(th06_rl_ranker PRIVATE cxx_std_20)
target_include_directories(th06_rl_ranker PUBLIC include)
target_compile_definitions(th06_rl_ranker PRIVATE TH06_RL_RANKER_BUILD=1)

if(MSVC)
  target_compile_options(th06_rl_native PRIVATE /W4 /permissive-)
  target_compile_options(th06_rl_ranker PRIVATE /W4 /permissive-)
else()
  target_compile_options(th06_rl_native PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(th06_rl_ranker PRIVATE -Wall -Wextra -Wpedantic)
endif()

# The Windows controller is also exercised under Wine with an embeddable
# 32-bit Python. Keep that validation DLL self-contained instead of requiring
# compiler-runtime DLLs to be copied into either the Python or retail-game
# directory.
if(MINGW)
  target_link_options(
    th06_rl_native PRIVATE -static -static-libgcc -static-libstdc++
  )
  target_link_options(
    th06_rl_ranker PRIVATE -static -static-libgcc -static-libstdc++
  )

  # The retail game remains a 32-bit executable, and the embeddable Wine
  # controller currently loads a 32-bit scorer.  GCC otherwise emits legacy
  # x87 arithmetic for this target.  Keep the fitted float32 model unchanged
  # while using the baseline SSE2 scalar ABI available on every supported
  # host; exact portable action conformance is still required before a DLL is
  # admitted by the policy.
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    target_compile_options(th06_rl_ranker PRIVATE -msse2 -mfpmath=sse)
  endif()
endif()
