// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lib_close.hpp"

#include <gtest/gtest.h>

#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::util;
auto get_tensor_name = [] -> const std::string {
    if (std::string(TEST_ENABLE_PIR) == "1") {
        return "3";
    } else {
        return "conv2d_0.tmp_0";
    }
};

INSTANTIATE_TEST_SUITE_P(
    Paddle,
    FrontendLibCloseTest,
    Values(std::make_tuple("paddle",
                           path_join({TEST_PADDLE_MODELS_DIRNAME, "conv2d_relu/conv2d_relu" + std::string(TEST_PADDLE_MODEL_EXT)}).string(),
                           get_tensor_name())),
    FrontendLibCloseTest::get_test_case_name);
