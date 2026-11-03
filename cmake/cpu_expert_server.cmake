# cmake/cpu_expert_server.cmake - the CPU-only remote-experts server (docs/REMOTE_EXPERTS.md, "CPU-only host").
#
# Included at the end of the top-level CMakeLists.txt.  With STRATA_ENABLE_CUDA=OFF (and no HIP) the tree builds no engine
# (the GPU engine is the `strata` target of the CUDA / HIP configure); this adds `strata-expert-server`, the one program of the
# engine that needs no GPU: it answers a conductor's expert requests with the CPU expert kernels over the model's experts, read
# from the GGUF in place.  It links no CUDA runtime and never initialises CUDA, so a host without a card (the T340) can serve.
#
#   cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF
#   cmake --build build-cpu --target strata-expert-server -j8
#
# The shared host sources (expert_source.cpp, net_experts.cpp) are the CUDA engine's own files, compiled unchanged: the few
# CUDA runtime calls they make resolve to include/strata/cpu_only/cuda_runtime.h (every device call fails with "no device",
# which they already handle), and the few GPU symbols they name resolve to src/core/cpu_only_stubs.cpp.
option(STRATA_BUILD_EXPERT_SERVER "Build strata-expert-server, the CPU-only remote-experts server (only without CUDA / HIP)" ON)

if(STRATA_BUILD_EXPERT_SERVER AND NOT STRATA_ENABLE_CUDA AND NOT STRATA_ENABLE_HIP AND NOT STRATA_ENABLE_SYCL
   AND UNIX AND STRATA_CPU_X86 AND TARGET strata_kernels_cpu AND STRATA_NATIVE_EXPERTS)
  add_library(strata_expert_host STATIC
    src/core/expert_source.cpp
    src/core/net_experts.cpp
    src/core/cpu_only_stubs.cpp
    src/platform/memory.cpp
    src/platform/io_priority.cpp)
  target_include_directories(strata_expert_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/cpu_only)
  target_compile_definitions(strata_expert_host PUBLIC STRATA_CPU_ONLY=1)
  target_link_libraries(strata_expert_host PUBLIC strata_kernels_cpu strata_artifact strata_warnings Threads::Threads)

  add_executable(strata-expert-server src/program/expert_server_main.cpp)
  target_link_libraries(strata-expert-server PRIVATE strata_expert_host)
  message(STATUS "Strata: CPU-only expert server enabled (strata-expert-server; no CUDA)")

  if(STRATA_BUILD_TESTS)
    # The binary starts, parses the engine's server command line, and refuses a missing model: no GPU, no CUDA library.
    add_test(NAME expert_server_cli COMMAND strata-expert-server --help)
    add_test(NAME expert_server_needs_model COMMAND strata-expert-server --expert-server --listen 7840)
    set_tests_properties(expert_server_needs_model PROPERTIES WILL_FAIL TRUE)
    # It must not link a CUDA library (the whole point of the target): readelf lists the NEEDED entries.
    add_test(NAME expert_server_links_no_cuda
             COMMAND ${CMAKE_COMMAND} -DBIN=$<TARGET_FILE:strata-expert-server> -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_no_cuda_link.cmake)
  endif()
endif()
