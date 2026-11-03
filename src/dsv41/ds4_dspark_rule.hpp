// Strata. The DeepSeek V4.1 engine's "--dspark auto" rule.  Header only, no GPU code (tests/dsv41/dspark_rule_test.cpp).
//
// The DSpark drafter makes decoding speculative: a block of drafts is checked by one verify window.  On one PC whose routed experts are
// far more than its GPUs hold (more than kDsparkAutoRatio x), the experts stream from the SSD, a verify window reads about 3.3x the
// bytes of one token and measured no faster than plain decoding, so "auto" leaves the drafter out there.  --dspark on / off / none says it outright.
#pragma once

namespace strata::ds4 {

constexpr double kDsparkAutoRatio = 1.5;

// experts_bytes: the routed experts in the pack; room_bytes: what the GPUs have free for experts; remote_active: --remote-experts is in use.
inline bool dspark_auto_keeps(double experts_bytes, double room_bytes, bool remote_active) {
    (void) remote_active;
    return !(experts_bytes > kDsparkAutoRatio * room_bytes);
}

}  // namespace strata::ds4
