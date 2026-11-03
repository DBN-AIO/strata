# cmake/dsv41_expert_server.cmake - the CPU-only DeepSeek-V4.1-Flash expert server (docs/DEEPSEEK41_SPLIT.md).
#
# Included at the end of the top-level CMakeLists.txt, next to cpu_expert_server.cmake and under the same condition: no CUDA / HIP /
# SYCL, x86, STRATA_NATIVE_EXPERTS.  It adds
#   strata-dsv41-expert-server   the server for a host without a card (T340) or without the CUDA toolkit yet (the stream PC)
#   ds4_net_test                 the synthetic split test's program (tools/dsv41/synthetic_split_test.py runs it)
# Both are made of src/dsv41/ds4_net.cpp (expert map, CPU expert host, conductor Remote), the protocol (net_experts.cpp, compiled
# through ds4_net_proto.cpp without the qwen38 pool) and the dsv41 engine's own CPU lane (ds4_gguf_cpu / ds4_iq_cpu / ds4_tq_cpu /
# ds4_gg_cpu + ggml-cpu): the same files strata-dsv41 uses, so the rows are the same bits.
#
#   cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF
#   cmake --build build-cpu --target strata-dsv41-expert-server ds4_net_test -j8
option(STRATA_BUILD_DSV41_EXPERT_SERVER "Build strata-dsv41-expert-server, the CPU-only DeepSeek expert server (only without CUDA / HIP)" ON)

if(STRATA_BUILD_DSV41_EXPERT_SERVER AND NOT STRATA_ENABLE_CUDA AND NOT STRATA_ENABLE_HIP AND NOT STRATA_ENABLE_SYCL
   AND (UNIX OR WIN32) AND STRATA_CPU_X86 AND TARGET strata_kernels_cpu AND STRATA_NATIVE_EXPERTS)
  find_package(Threads REQUIRED)
  add_library(strata_dsv41_expert_host STATIC
    src/dsv41/ds4_net.cpp
    src/dsv41/ds4_net_proto.cpp
    src/dsv41/ds4_gguf_cpu.cpp
    src/dsv41/ds4_iq_cpu.cpp
    src/dsv41/ds4_tq_cpu.cpp
    src/dsv41/ds4_gg_cpu.cpp
    src/dsv41/vendor/exl3_cpu.cpp)
  target_include_directories(strata_dsv41_expert_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
  target_compile_definitions(strata_dsv41_expert_host PUBLIC STRATA_DSV41_TQ_CPU=1)
  target_link_libraries(strata_dsv41_expert_host PUBLIC strata_kernels_cpu Threads::Threads)
  if(WIN32)
    target_link_libraries(strata_dsv41_expert_host PUBLIC ws2_32)
  endif()

  add_executable(strata-dsv41-expert-server src/dsv41/ds4_expert_server_main.cpp)
  target_link_libraries(strata-dsv41-expert-server PRIVATE strata_dsv41_expert_host)

  add_executable(ds4_net_test tests/dsv41/ds4_net_test.cpp)
  target_link_libraries(ds4_net_test PRIVATE strata_dsv41_expert_host)
  message(STATUS "Strata: CPU-only DeepSeek expert server enabled (strata-dsv41-expert-server; no CUDA)")
endif()
