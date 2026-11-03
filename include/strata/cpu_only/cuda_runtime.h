// include/strata/cpu_only/cuda_runtime.h - the CUDA runtime API as the CPU-only build sees it (STRATA_ENABLE_CUDA=OFF).
//
// The host sources the expert server shares with the CUDA engine (expert_source.cpp, on_device.hpp) ask the CUDA runtime
// for page-locked memory, events and streams.  A CPU-only build has no CUDA toolkit, so this header stands in for
// <cuda_runtime.h> (CMake puts this folder first on the include path of the CPU-only targets only).  Every call that would
// touch a device fails with cudaErrorNoDevice, which is the answer those sources already handle: "page-locking refused",
// "no device alias", "pageable it is".  Nothing here initialises CUDA, loads a driver or links libcudart.
//
// It declares only what the shared host sources use.  A new CUDA call in them needs a line here, or the CPU-only
// build (CI: .github/workflows/build-linux.yml) fails to compile - which is the intended alarm.
#pragma once
#include <cstddef>

#if defined(STRATA_CPU_ONLY)
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

enum {
    cudaSuccess = 0,
    cudaErrorNotReady = 600,
    cudaErrorNoDevice = 100,
};
enum { cudaHostAllocDefault = 0, cudaHostAllocPortable = 1, cudaHostAllocMapped = 2 };
enum { cudaHostRegisterDefault = 0, cudaHostRegisterPortable = 1, cudaHostRegisterMapped = 2 };
enum { cudaEventDefault = 0, cudaEventDisableTiming = 2 };
enum { cudaStreamDefault = 0, cudaStreamNonBlocking = 1 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
enum cudaDeviceAttr { cudaDevAttrIntegrated = 18 };

inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "no CUDA device (CPU-only build)"; }
inline cudaError_t cudaGetDevice(int* d) { if (d) *d = 0; return cudaErrorNoDevice; }
inline cudaError_t cudaSetDevice(int) { return cudaErrorNoDevice; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr, int) { if (v) *v = 0; return cudaErrorNoDevice; }
inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) { if (f) *f = 0; if (t) *t = 0; return cudaErrorNoDevice; }
inline cudaError_t cudaMalloc(void** p, size_t) { if (p) *p = nullptr; return cudaErrorNoDevice; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaHostAlloc(void** p, size_t, unsigned) { if (p) *p = nullptr; return cudaErrorNoDevice; }
inline cudaError_t cudaFreeHost(void*) { return cudaSuccess; }
inline cudaError_t cudaHostRegister(void*, size_t, unsigned) { return cudaErrorNoDevice; }
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }
inline cudaError_t cudaHostGetDevicePointer(void** p, void*, unsigned) { if (p) *p = nullptr; return cudaErrorNoDevice; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) { return cudaErrorNoDevice; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t = nullptr) { return cudaErrorNoDevice; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t = nullptr) { return cudaErrorNoDevice; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { if (s) *s = nullptr; return cudaErrorNoDevice; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { if (e) *e = nullptr; return cudaErrorNoDevice; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return cudaErrorNoDevice; }
inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
#else
#error "strata/cpu_only/cuda_runtime.h is for the CPU-only build (STRATA_CPU_ONLY); it must not shadow a real CUDA toolkit"
#endif
