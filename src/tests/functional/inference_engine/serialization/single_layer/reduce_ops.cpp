// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/reduce_ops.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(ReduceOpsLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8,
};

const std::vector<bool> keepDims = {
        true,
        false,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
        ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::Prod,
        ngraph::helpers::ReductionType::L1,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<ngraph::helpers::ReductionType> reductionLogicalTypes = {
        ngraph::helpers::ReductionType::LogicalOr,
        ngraph::helpers::ReductionType::LogicalAnd
};

const std::vector<std::vector<size_t>> inputShapesOneAxis = {
        std::vector<size_t>{10, 20, 30, 40},
        std::vector<size_t>{3, 5, 7, 9},
        std::vector<size_t>{10},
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 20, 30, 40},
        std::vector<size_t>{3, 5, 7, 9},
};

const std::vector<std::vector<int>> axes = {
        {0},
        {1},
        {2},
        {3},
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3},
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3},
        {0, 1, 2, 3},
        {1, -1}
};

const auto paramsOneAxis = testing::Combine(
        testing::Values(std::vector<int>{0}),
        testing::ValuesIn(opTypes),
        testing::ValuesIn(keepDims),
        testing::ValuesIn(reductionTypes),
        testing::Values(netPrecisions[0]),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapesOneAxis),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto paramsOneAxisLogical = testing::Combine(
        testing::Values(std::vector<int>{0}),
        testing::ValuesIn(opTypes),
        testing::ValuesIn(keepDims),
        testing::ValuesIn(reductionLogicalTypes),
        testing::Values(InferenceEngine::Precision::BOOL),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapesOneAxis),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto params_Axes = testing::Combine(
        testing::ValuesIn(axes),
        testing::Values(opTypes[1]),
        testing::ValuesIn(keepDims),
        testing::Values(reductionTypes[0]),
        testing::Values(netPrecisions[0]),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapes),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto params_ReductionTypes = testing::Combine(
        testing::Values(std::vector<int>{0, 1, 3}),
        testing::Values(opTypes[1]),
        testing::ValuesIn(keepDims),
        testing::ValuesIn(reductionTypes),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{2, 9, 2, 9}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto params_ReductionTypesLogical = testing::Combine(
        testing::Values(std::vector<int>{0, 1, 3}),
        testing::Values(opTypes[1]),
        testing::ValuesIn(keepDims),
        testing::ValuesIn(reductionLogicalTypes),
        testing::Values(InferenceEngine::Precision::BOOL),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{2, 9, 2, 9}),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceOneAxis_Serialization,
        ReduceOpsLayerTest,
        paramsOneAxis,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceLogicalOneAxis_Serialization,
        ReduceOpsLayerTest,
        paramsOneAxisLogical,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceAxes_Serialization,
        ReduceOpsLayerTest,
        params_Axes,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_ReductionTypes_Serialization,
        ReduceOpsLayerTest,
        params_ReductionTypes,
        ReduceOpsLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceLogical_ReductionTypes_Serialization,
        ReduceOpsLayerTest,
        params_ReductionTypesLogical,
        ReduceOpsLayerTest::getTestCaseName
);
}   // namespace