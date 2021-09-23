// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/config.hpp"
#include "cldnn/cldnn_config.hpp"
#include "gpu/gpu_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    IE_SUPPRESS_DEPRECATED_START
    const std::vector<std::map<std::string, std::string>> inconfigs = {
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
            {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, "OFF"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, "ON"}},
            {{InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}},
            {{InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}
    };

    const std::vector<std::map<std::string, std::string>> multiinconfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}
    };


    const std::vector<std::map<std::string, std::string>> autoinconfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}
    };
    IE_SUPPRESS_DEPRECATED_END

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                ::testing::ValuesIn(inconfigs)),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(multiinconfigs)),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(autoinconfigs)),
            IncorrectConfigTests::getTestCaseName);


    const std::vector<std::map<std::string, std::string>> conf = {
            {}
    };

    IE_SUPPRESS_DEPRECATED_START
    const std::vector<std::map<std::string, std::string>> conf_gpu = {
            // Deprecated
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE, "0"}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE, "1"}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY, "0"}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY, "1"}},

            {{InferenceEngine::GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE, "0"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE, "1"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY, "0"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY, "1"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_MAX_NUM_THREADS, "1"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_MAX_NUM_THREADS, "4"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
    };
    IE_SUPPRESS_DEPRECATED_END

    const std::vector<std::map<std::string, std::string>> multiconf = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}}
    };

    const std::vector<std::map<std::string, std::string>> autoConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}}
    };


    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigAPITests,
            ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                ::testing::ValuesIn(conf)),
            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GPU_BehaviorTests, CorrectConfigAPITests,
            ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                ::testing::ValuesIn(conf_gpu)),
            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, CorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                    ::testing::ValuesIn(multiconf)),
            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, CorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                    ::testing::ValuesIn(autoConfigs)),
            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                    ::testing::ValuesIn(inconfigs)),
             IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                    ::testing::ValuesIn(multiinconfigs)),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::ValuesIn(netPrecisions),
                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                    ::testing::ValuesIn(autoinconfigs)),
            IncorrectConfigAPITests::getTestCaseName);


} // namespace
