// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <atomic>
#include <iomanip>

#include <vpu/utils/io.hpp>
#include <vpu/private_plugin_config.hpp>
#include <vpu/configuration/options/log_level.hpp>
#include <vpu/configuration/options/copy_optimization.hpp>
#include <vpu/configuration/options/protocol.hpp>
#include <vpu/configuration/options/power_config.hpp>
#include <vpu/configuration/options/hw_acceleration.hpp>
#include <vpu/configuration/options/hw_extra_split.hpp>
#include <vpu/configuration/options/hw_pool_conv_merge.hpp>
#include <vpu/configuration/options/hw_black_list.hpp>
#include <vpu/configuration/options/hw_inject_stages.hpp>
#include <vpu/configuration/options/hw_dilation.hpp>
#include <vpu/configuration/options/tiling_cmx_limit_kb.hpp>
#include <vpu/configuration/options/watchdog_interval.hpp>
#include <vpu/configuration/options/enable_receiving_tensor_time.hpp>
#include <vpu/configuration/options/perf_report_mode.hpp>
#include <vpu/configuration/options/perf_count.hpp>
#include <vpu/configuration/options/pack_data_in_cmx.hpp>
#include <vpu/configuration/options/number_of_shaves.hpp>
#include <vpu/configuration/options/number_of_cmx_slices.hpp>
#include <vpu/configuration/options/throughput_streams.hpp>
#include <vpu/configuration/options/vpu_scales_option.hpp>
#include <vpu/configuration/options/tensor_strides.hpp>
#include <vpu/configuration/options/ignore_unknown_layers.hpp>
#include <vpu/configuration/options/force_pure_tensor_iterator.hpp>
#include <vpu/configuration/options/enable_tensor_iterator_unrolling.hpp>
#include <vpu/configuration/options/exclusive_async_requests.hpp>
#include <vpu/configuration/options/enable_weights_analysis.hpp>
#include <vpu/configuration/options/enable_repl_with_screlu.hpp>
#include <vpu/configuration/options/enable_permute_merging.hpp>
#include <vpu/configuration/options/enable_memory_types_annotation.hpp>
#include <vpu/configuration/options/dump_internal_graph_file_name.hpp>
#include <vpu/configuration/options/dump_all_passes_directory.hpp>
#include <vpu/configuration/options/dump_all_passes.hpp>
#include <vpu/configuration/options/disable_convert_stages.hpp>
#include <vpu/configuration/options/disable_reorder.hpp>
#include <vpu/configuration/options/device_id.hpp>
#include <vpu/configuration/options/device_connect_timeout.hpp>
#include <vpu/configuration/options/detect_network_batch.hpp>
#include <vpu/configuration/options/custom_layers.hpp>
#include <vpu/configuration/options/config_file.hpp>
#include <vpu/configuration/options/memory_type.hpp>
#include <vpu/configuration/options/enable_force_reset.hpp>
#include <vpu/configuration/options/check_preprocessing_inside_model.hpp>
#include <vpu/configuration/options/enable_early_eltwise_relu_fusion.hpp>
#include <vpu/configuration/options/enable_custom_reshape_param.hpp>
#include <vpu/configuration/options/none_layers.hpp>
#include <vpu/configuration/options/enable_async_dma.hpp>
#include "vpu/configuration/options/performance_hint.hpp"
#include "vpu/configuration/options/performance_hint_num_requests.hpp"
#include "vpu/configuration/options/ov_throughput_streams.hpp"

namespace vpu {

StagePtr TestStage::cloneImpl() const {
    return std::make_shared<TestStage>(*this);
}

namespace {

template <typename Value>
void setInOutPortInfo(
        const Stage& stage,
        const std::string& attrBaseName,
        StageDataInfo<Value>& info) {
    auto inAttrName = formatString("test_input_%s_info", attrBaseName);
    auto outAttrName = formatString("test_output_%s_info", attrBaseName);

    if (stage->attrs().has(inAttrName)) {
        const auto& inputInfo = stage->attrs().get<InOutPortMap<Value>>(inAttrName);

        for (const auto& p : inputInfo) {
            info.setInput(stage->inputEdge(p.first), p.second);
        }
    }

    if (stage->attrs().has(outAttrName)) {
        const auto& outputInfo = stage->attrs().get<InOutPortMap<Value>>(outAttrName);

        for (const auto& p : outputInfo) {
            info.setOutput(stage->outputEdge(p.first), p.second);
        }
    }
}

}

void TestStage::propagateDataOrderImpl(StageDataInfo<DimsOrder>& orderInfo) {
    setInOutPortInfo(this, "DataOrder", orderInfo);
}

void TestStage::getDataStridesRequirementsImpl(StageDataInfo<StridesRequirement>& stridesInfo) {
    setInOutPortInfo(this, "Strides", stridesInfo);
}

void TestStage::finalizeDataLayoutImpl() {
}

void TestStage::getBatchSupportInfoImpl(StageDataInfo<BatchSupport>& batchInfo) {
    setInOutPortInfo(this, "Batch", batchInfo);

    if (attrs().has("test_input_Batch_info")) {
        for (const auto& outEdge : outputEdges()) {
            batchInfo.setOutput(outEdge, BatchSupport::Split);
        }
    }
}

void TestStage::serializeParamsImpl(BlobSerializer&) const {
}

void TestStage::serializeDataImpl(BlobSerializer&) const {
}

TestModel::TestModel(const Model& model, const DataDesc& dataDesc) :
        _model(model), _dataDesc(dataDesc) {
}

void TestModel::createInputs(int numInputs) {
    _model->attrs().set<int>("numInputs", numInputs);
    _inputs.resize(numInputs);

    for (int i = 0; i < numInputs; ++i) {
        _inputs[i] = _model->addInputData(formatString("Input %d", i), _dataDesc);
    }
}

void TestModel::createOutputs(int numOutputs) {
    _model->attrs().set<int>("numOutputs", numOutputs);
    _outputs.resize(numOutputs);

    for (int i = 0; i < numOutputs; ++i) {
        _outputs[i] = _model->addOutputData(formatString("Output %d", i), _dataDesc);
    }
}

int TestModel::addStage(
        std::initializer_list<InputInfo> curInputInfos,
        std::initializer_list<OutputInfo> curOutputInfos) {
    DataVector curInputs;
    for (const auto& info : curInputInfos) {
        if (info.type == InputType::Original) {
            curInputs.push_back(_inputs.at(info.originalInputInd));
        } else {
            curInputs.push_back(_stages.at(info.prevStageInd)->output(info.prevStageOutputInd));
        }
    }

    DataVector curOutputs;
    for (const auto& info : curOutputInfos) {
        if (info.type == OutputType::Original) {
            curOutputs.push_back(_outputs.at(info.originalOutputInd));
        } else {
            curOutputs.push_back(_model->addNewData(formatString("Data %d / %d", _stages.size(), curOutputs.size()), _dataDesc));
        }
    }

    auto stage = _model->addNewStage<TestStage>(
        formatString("Stage %m%m%d", std::setw(2), std::setfill('0'), _stages.size()),
        StageType::None,
        nullptr,
        curInputs,
        curOutputs);
    stage->attrs().set<int>("test_ind", _stages.size());

    _stages.push_back(stage);

    return _stages.size() - 1;
}
void TestModel::setStageDataOrderInfo(
        int stageInd,
        const InOutPortMap<DimsOrder>& inputInfo,
        const InOutPortMap<DimsOrder>& outputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_DataOrder_info", inputInfo);
    }
    if (!outputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_DataOrder_info", outputInfo);
    }
}

void TestModel::setStageStridesInfo(
        int stageInd,
        const InOutPortMap<StridesRequirement>& inputInfo,
        const InOutPortMap<StridesRequirement>& outputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Strides_info", inputInfo);
    }
    if (!outputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Strides_info", outputInfo);
    }
}

void TestModel::setStageBatchInfo(
        int stageInd,
        const InOutPortMap<BatchSupport>& inputInfo) {
    if (!inputInfo.empty()) {
        _stages.at(stageInd)->attrs().set("test_input_Batch_info", inputInfo);
    }
}

PluginConfiguration createConfiguration() {
    PluginConfiguration configuration;
    configuration.registerOption<LogLevelOption>();
    configuration.registerOption<CopyOptimizationOption>();
    configuration.registerOption<ProtocolOption>();
    configuration.registerOption<PowerConfigOption>();
    configuration.registerOption<HwAccelerationOption>();
    configuration.registerOption<HwExtraSplitOption>();
    configuration.registerOption<HwPoolConvMergeOption>();
    configuration.registerOption<HwBlackListOption>();
    configuration.registerOption<HwInjectStagesOption>();
    configuration.registerOption<HwDilationOption>();
    configuration.registerOption<TilingCMXLimitKBOption>();
    configuration.registerOption<WatchdogIntervalOption>();
    configuration.registerOption<EnableReceivingTensorTimeOption>();
    configuration.registerOption<PerfReportModeOption>();
    configuration.registerOption<PerfCountOption>();
    configuration.registerOption<PackDataInCMXOption>();
    configuration.registerOption<NumberOfSHAVEsOption>();
    configuration.registerOption<NumberOfCMXSlicesOption>();
    configuration.registerOption<ThroughputStreamsOption>();
    configuration.registerOption<VPUScalesOption>();
    configuration.registerOption<TensorStridesOption>();
    configuration.registerOption<IgnoreUnknownLayersOption>();
    configuration.registerOption<ForcePureTensorIteratorOption>();
    configuration.registerOption<EnableTensorIteratorUnrollingOption>();
    configuration.registerOption<ExclusiveAsyncRequestsOption>();
    configuration.registerOption<EnableWeightsAnalysisOption>();
    configuration.registerOption<EnableReplWithSCReluOption>();
    configuration.registerOption<EnablePermuteMergingOption>();
    configuration.registerOption<EnableMemoryTypesAnnotationOption>();
    configuration.registerOption<DumpInternalGraphFileNameOption>();
    configuration.registerOption<DumpAllPassesDirectoryOption>();
    configuration.registerOption<DumpAllPassesOption>();
    configuration.registerOption<DeviceIDOption>();
    configuration.registerOption<DeviceConnectTimeoutOption>();
    configuration.registerOption<DetectNetworkBatchOption>();
    configuration.registerOption<CustomLayersOption>();
    configuration.registerOption<ConfigFileOption>();
    configuration.registerOption<MemoryTypeOption>();
    configuration.registerOption<EnableForceResetOption>();
    configuration.registerOption<CheckPreprocessingInsideModelOption>();
    configuration.registerOption<EnableEarlyEltwiseReluFusionOption>();
    configuration.registerOption<EnableCustomReshapeParamOption>();
    configuration.registerOption<NoneLayersOption>();
    configuration.registerOption<EnableAsyncDMAOption>();
    configuration.registerOption<PerformanceHintOption>();
    configuration.registerOption<PerformanceHintNumRequestsOption>();
    configuration.registerOption<OvThroughputStreamsOption>();
IE_SUPPRESS_DEPRECATED_START
    configuration.registerDeprecatedOption<DisableConvertStagesOption>(InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES);
    configuration.registerDeprecatedOption<DisableReorderOption>(InferenceEngine::MYRIAD_DISABLE_REORDER);
IE_SUPPRESS_DEPRECATED_END
    return configuration;
}

void GraphTransformerTest::SetUp() {
    ASSERT_NO_FATAL_FAILURE(TestsCommon::SetUp());

    _log = std::make_shared<Logger>(
        "Test",
        LogLevel::Debug,
        consoleOutput());

    stageBuilder = std::make_shared<StageBuilder>();
    frontEnd = std::make_shared<FrontEnd>(stageBuilder, _mockCore);
    backEnd = std::make_shared<BackEnd>();
    passManager = std::make_shared<PassManager>(stageBuilder, backEnd);

    config = createConfiguration();
}

void GraphTransformerTest::TearDown() {
    for (const auto& model : _models) {
        backEnd->dumpModel(model);
    }

    if (compileEnvInitialized) {
        CompileEnv::free();
    }

    TestsCommon::TearDown();
}

void GraphTransformerTest::InitCompileEnv() {
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_FILE_NAME")) {
        config.set(InferenceEngine::MYRIAD_DUMP_INTERNAL_GRAPH_FILE_NAME, envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_DIRECTORY")) {
        config.set(InferenceEngine::MYRIAD_DUMP_ALL_PASSES_DIRECTORY, envVar);
    }
    if (const auto envVar = std::getenv("IE_VPU_DUMP_ALL_PASSES")) {
        config.set(InferenceEngine::MYRIAD_DUMP_ALL_PASSES, std::stoi(envVar) != 0
            ? InferenceEngine::PluginConfigParams::YES : InferenceEngine::PluginConfigParams::NO);
    }

    CompileEnv::init(config, _log);
    compileEnvInitialized = true;
}

namespace {

std::atomic<int> g_counter(0);

}

Model GraphTransformerTest::CreateModel() {
    const auto& env = CompileEnv::get();

    auto unitTest = testing::UnitTest::GetInstance();
    IE_ASSERT(unitTest != nullptr);
    auto curTestInfo = unitTest->current_test_info();
    IE_ASSERT(curTestInfo != nullptr);

    auto model = std::make_shared<ModelObj>(
        formatString("%s/%s", curTestInfo->test_case_name(), curTestInfo->name()));
    model->attrs().set<int>("index", g_counter.fetch_add(1));
    model->attrs().set<Resources>("resources", env.resources);

    _models.push_back(model);

    return model;
}

TestModel GraphTransformerTest::CreateTestModel(const DataDesc& dataDesc) {
    return TestModel(CreateModel(), dataDesc);
}

}
