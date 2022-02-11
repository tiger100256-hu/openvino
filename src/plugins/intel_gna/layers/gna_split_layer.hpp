// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <legacy/ie_layers.h>
#include "backend/gna_limitations.hpp"

namespace GNAPluginNS {
// Split, Slice
class GNASplitLayer {
    InferenceEngine::CNNLayerPtr splitLayer;

public:
    explicit GNASplitLayer(InferenceEngine::CNNLayerPtr layer) :
        splitLayer(layer)
    {}

    InferenceEngine::CNNLayerPtr getSplit() { return splitLayer; }
    /**
     * gna memory of this size is reserved for split
     */
    size_t reserved_size = 0;
    bool output_allocation_flag = false;
    /**
     * gna memory of this offset from gna_ptr
     */
    struct SplitConnectedLayerInfo {
        SplitConnectedLayerInfo() = default;
        SplitConnectedLayerInfo(InferenceEngine::CNNLayerPtr connectedTo,
            int insDataIdx,
            size_t o,
            size_t p) :
            connectedTo(connectedTo),
            insDataIdx(insDataIdx),
            offset(o),
            pure_size(p) {}

        InferenceEngine::CNNLayerPtr  connectedTo;
        int          insDataIdx = 0;
        size_t       offset = 0;  // in number of elements of input layer
        size_t       pure_size = 0;
    };
    std::vector<SplitConnectedLayerInfo> splitOutputLayers;
};

// @brief Returns sizes of split outputs to split the input tensor to aligned parts not greater than the specified size
static std::vector<uint32_t> GetAlignedSplitSizes(uint32_t totalSize, uint32_t maxSplitSize, uint32_t alignment = GNALimitations::inputByteAlignment) {
    std::vector<uint32_t> splitSizes;
    uint32_t maxAlignedSplitSize = maxSplitSize - maxSplitSize % alignment;
    uint32_t usedSize = 0;
    while (usedSize < totalSize) {
        uint32_t partSize = std::min(totalSize - usedSize, maxAlignedSplitSize);
        splitSizes.push_back(partSize);
        usedSize += partSize;
    }
    return splitSizes;
}

}  // namespace GNAPluginNS
