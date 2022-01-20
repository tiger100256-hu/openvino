// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_nonzero.hpp"
#include "vpu/private_plugin_config.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
    {}
};

const std::vector<std::map<std::string, std::string>> importConfigs = {
    {}
};

const std::vector<std::string> appHeaders = {
        "",
        "APPLICATION_HEADER"
};

std::vector<size_t> inputShape = ngraph::Shape{1000};

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkCase, ImportNonZero,
                        ::testing::Combine(
                            ::testing::Values(inputShape),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigs),
                            ::testing::ValuesIn(appHeaders)),
                        ImportNonZero::getTestCaseName);

} // namespace
