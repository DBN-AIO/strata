# cmake/strata_cuda_arch_check.cmake - refuse a CUDA architecture list the toolkit cannot build (BBXL, docs/DEEPSEEK41_SPLIT.md
# "Blackwell (sm_120) builds").  Included by the top-level CMakeLists.txt after enable_language(CUDA); also runnable without nvcc:
#   cmake -DCHECK_ARCHS="86-real;120" -DCHECK_NVCC_VERSION=12.4 -P cmake/strata_cuda_arch_check.cmake      (exit 1 + message = refused)
# The same table is tools/dsv41/cuda_arch_check.py (the build scripts call that one); tests/dsv41/test_cuda_arch_check.py runs both.
#   sm_100 / sm_101 / sm_120  (Blackwell)  need nvcc 12.8 or newer
#   sm_121 (GB10)                          needs nvcc 12.9 or newer
function(strata_cuda_arch_check archs nvcc_version)
  foreach(_a IN LISTS archs)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    if(NOT _base MATCHES "^[0-9]+$")
      continue()                                      # native / all / all-major: the toolkit decides
    endif()
    set(_need "")
    if(_base GREATER_EQUAL 121)
      set(_need 12.9)
    elseif(_base GREATER_EQUAL 100)
      set(_need 12.8)
    endif()
    if(_need AND nvcc_version VERSION_LESS _need)
      message(FATAL_ERROR
        "CMAKE_CUDA_ARCHITECTURES '${_a}' (sm_${_base}) needs nvcc ${_need} or newer, but this toolkit is ${nvcc_version}. "
        "Install CUDA ${_need}+ (13.0 is the release toolkit) and pass -DCMAKE_CUDA_COMPILER=<toolkit>/bin/nvcc, "
        "or build for the older architectures only (-DCMAKE_CUDA_ARCHITECTURES=86-real). docs/DEEPSEEK41_SPLIT.md, \"Blackwell (sm_120) builds\".")
    endif()
  endforeach()
endfunction()

if(DEFINED CHECK_ARCHS AND DEFINED CHECK_NVCC_VERSION)
  strata_cuda_arch_check("${CHECK_ARCHS}" "${CHECK_NVCC_VERSION}")
  message(STATUS "arch ${CHECK_ARCHS} ok for nvcc ${CHECK_NVCC_VERSION}")
endif()
