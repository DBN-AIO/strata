// src/program/q36_expert_server.hpp - the qwen35moe half of strata-expert-server (q36_expert_server.cpp).
#pragma once

#include "strata/core/geometry.hpp"

#include <cstdint>
#include <string>

namespace strata_q36 {

struct Options {
    std::string native;          ///< shard 1 of the model
    std::string listen_host;     ///< empty: 127.0.0.1
    int listen_port = 7840;
    std::string serve_layers;    ///< "a-b" (inclusive; the MTP block is index n_layers); empty: every block
    int pool_workers = 0;
    bool no_host_worker = false;
};

/// Repack the served blocks' experts into RAM and answer a conductor's requests until stopped.  Returns the exit code.
int serve(const Options& o, const strata::core::ModelGeometry& g, int64_t top_k);

}  // namespace strata_q36
