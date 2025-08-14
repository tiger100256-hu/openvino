// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> result =
    {
#ifdef OPENVINO_STATIC_LIBRARY
        // Disable tests for static libraries
        ".*FrontendLibCloseTest.*",
#endif
        ".*testUnloadLibBeforeDeletingDependentObject.*",
    };
    if (TEST_PADDLE_MODEL_EXT == ".json") {
        result.insert(result.end(),
        {
            "Paddle_Places.*",
        });
    }
    return result;
}
