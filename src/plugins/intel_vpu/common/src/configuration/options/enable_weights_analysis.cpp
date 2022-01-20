// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_weights_analysis.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void EnableWeightsAnalysisOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableWeightsAnalysisOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableWeightsAnalysisOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_WEIGHTS_ANALYSIS;
}

details::Access EnableWeightsAnalysisOption::access() {
    return details::Access::Private;
}

details::Category EnableWeightsAnalysisOption::category() {
    return details::Category::CompileTime;
}

std::string EnableWeightsAnalysisOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

EnableWeightsAnalysisOption::value_type EnableWeightsAnalysisOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
