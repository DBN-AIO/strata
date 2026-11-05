// Strata. strata-dsv41-expert-server: the DeepSeek-V4.1-Flash expert server with no CUDA in it (docs/DEEPSEEK41_SPLIT.md).
#include "ds4_net.hpp"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--version") == 0) {
#ifdef DS4_BUILD_ID
        std::printf("strata-dsv41-expert-server %s\n", DS4_BUILD_ID);
#else
        std::printf("strata-dsv41-expert-server\n");
#endif
        return 0;
    }
    return strata::ds4::net::server_main(argc, argv);
}
