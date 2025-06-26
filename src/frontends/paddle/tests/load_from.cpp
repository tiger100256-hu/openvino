// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "load_from.hpp"

#include "paddle_utils.hpp"

using namespace ov::frontend;

using PaddleCutTest = FrontEndLoadFromTest;

static LoadFromFEParam getTestData() {
    LoadFromFEParam res;
    res.m_frontEndName = PADDLE_FE;
    res.m_modelsPath = std::string(TEST_PADDLE_MODELS_DIRNAME);
    res.m_file = "conv2d/conv2d" + std::string(TEST_PADDLE_MODEL_EXT);
    res.m_files = {"2in_2out/2in_2out" + std::string(TEST_PADDLE_MODEL_EXT), "2in_2out/2in_2out.pdiparams"};
    res.m_stream = "relu/relu" + std::string(TEST_PADDLE_MODEL_EXT);
    res.m_streams = {"2in_2out/2in_2out" + std::string(TEST_PADDLE_MODEL_EXT), "2in_2out/2in_2out.pdiparams"};
    return res;
}

INSTANTIATE_TEST_SUITE_P(PaddleCutTest,
                         FrontEndLoadFromTest,
                         ::testing::Values(getTestData()),
                         FrontEndLoadFromTest::getTestCaseName);
