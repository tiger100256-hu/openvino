// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "basic_api.hpp"

#include "paddle_utils.hpp"

using namespace ov::frontend;

using PaddleBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models{
    std::string("conv2d/conv2d" + std::string(TEST_PADDLE_MODEL_EXT)),
    std::string("conv2d_relu/conv2d_relu" + std::string(TEST_PADDLE_MODEL_EXT)),
    std::string("2in_2out/2in_2out" + std::string(TEST_PADDLE_MODEL_EXT)),
    std::string("multi_tensor_split/multi_tensor_split" + std::string(TEST_PADDLE_MODEL_EXT)),
    std::string("2in_2out_dynbatch/2in_2out_dynbatch" + std::string(TEST_PADDLE_MODEL_EXT)),
};

INSTANTIATE_TEST_SUITE_P(PaddleBasicTest,
                         FrontEndBasicTest,
                         ::testing::Combine(::testing::Values(PADDLE_FE),
                                            ::testing::Values(std::string(TEST_PADDLE_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         FrontEndBasicTest::getTestCaseName);
