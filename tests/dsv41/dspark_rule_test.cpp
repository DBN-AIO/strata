// Strata. The "--dspark auto" rule (src/dsv41/ds4_dspark_rule.hpp).  No GPU needed.
#include "ds4_dspark_rule.hpp"

#include <cstdio>

using strata::ds4::dspark_auto_keeps;

static int fails = 0;
#define CHECK(c)                                                  \
    do {                                                          \
        if (!(c)) {                                               \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                              \
        }                                                         \
    } while (0)

int main() {
    const double GB = 1e9;
    // one PC (no remote experts): unchanged.  132 GB of experts on a 35 GB card streams from the SSD: no drafter; 40 GB on 35 fits: drafter
    CHECK(!dspark_auto_keeps(132 * GB, 35 * GB, false));
    CHECK(dspark_auto_keeps(40 * GB, 35 * GB, false));
    CHECK(dspark_auto_keeps(52.5 * GB, 35 * GB, false));   // exactly 1.5x: kept
    CHECK(!dspark_auto_keeps(52.6 * GB, 35 * GB, false));
    CHECK(!dspark_auto_keeps(10 * GB, 0, false));
    // --remote-experts: a verify window is ONE batched call per served layer (not k single calls), so the drafter loads whatever the
    // experts weigh against the local GPUs: the plan in docs/DEEPSEEK41_SPLIT.md (132 GB of experts, 35 GB of room) is exactly this case
    CHECK(dspark_auto_keeps(132 * GB, 35 * GB, true));
    CHECK(dspark_auto_keeps(208 * GB, 35 * GB, true));
    CHECK(dspark_auto_keeps(132 * GB, 0, true));
    std::printf(fails ? "dspark_rule_test: %d failure(s)\n" : "dspark_rule_test: ok\n", fails);
    return fails ? 1 : 0;
}
