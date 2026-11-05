// Strata. The remote-experts protocol (src/core/net_experts.cpp) without the qwen38 engine's expert pool: the dsv41 programs
// (strata-dsv41, strata-dsv41-expert-server) take the client, the generic server loop and the SHA/HMAC from it.  A wrapper
// translation unit, so the define does not leak into the targets that compile net_experts.cpp whole (strata_engine, the
// qwen38 CPU-only server).
#define STRATA_NET_NO_QWEN 1
#include "../core/net_experts.cpp"
