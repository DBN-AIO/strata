// include/strata/core/net_random.hpp - random bytes from the operating system, for the handshake nonce (net_experts.cpp).
//
// Not std::random_device: on some MinGW builds it is a fixed pseudo-random sequence, which would make the server's nonce
// predictable and a recorded HELLO replayable.  Linux: getrandom(2) (/dev/urandom when the kernel lacks it), macOS:
// arc4random_buf, Windows: BCryptGenRandom with the system-preferred RNG (link bcrypt).  False on any failure: the caller
// refuses the connection, there is no weaker fallback.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <bcrypt.h>
#elif defined(__APPLE__)
#  include <cstdlib>
#else
#  include <cerrno>
#  include <unistd.h>
#  if defined(__linux__)
#    include <sys/random.h>
#  endif
#endif

namespace strata::core::netproto {

inline bool os_random(uint8_t* p, size_t n) {
#if defined(_WIN32)
    while (n > 0) {
        const ULONG c = n > (1u << 20) ? (1u << 20) : (ULONG) n;
        if (BCryptGenRandom(nullptr, p, c, BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) return false;   // (NTSTATUS < 0: failure)
        p += c;
        n -= c;
    }
    return true;
#elif defined(__APPLE__)
    arc4random_buf(p, n);
    return true;
#else
#  if defined(__linux__)
    while (n > 0) {
        const ssize_t r = ::getrandom(p, n, 0);
        if (r > 0) { p += r; n -= (size_t) r; continue; }
        if (r < 0 && errno == EINTR) continue;
        break;   // ENOSYS (an old kernel) or another failure: try the device below for the rest
    }
    if (n == 0) return true;
#  endif
    std::FILE* f = std::fopen("/dev/urandom", "rb");
    if (f == nullptr) return false;
    const size_t got = std::fread(p, 1, n, f);
    std::fclose(f);
    return got == n;
#endif
}

}  // namespace strata::core::netproto
