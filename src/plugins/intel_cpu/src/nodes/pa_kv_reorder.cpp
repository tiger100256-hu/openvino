// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "transformations/cpu_opset/common/op/pa_kv_reorder.hpp"

namespace ov::intel_cpu::node {

bool PaKVReorder::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::as_type_ptr<const ov::intel_cpu::PaKVReorderNode>(op)) {
            errorMessage = "Only PaKVReorder operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

PaKVReorder::PaKVReorder(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

void PaKVReorder::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto in_precs = getOriginalInputPrecisions();
    auto out_precs = getOriginalOutputPrecisions();

    in_precs[2] = ov::element::i32;
    in_precs[3] = ov::element::i32;
    in_precs[4] = ov::element::i32;
    in_precs[5] = ov::element::i32;
    out_precs[0] = ov::element::u8;

    std::vector<PortConfigurator> in_configs;
    in_configs.reserve(getOriginalInputsNumber());
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        in_configs.emplace_back(LayoutType::ncsp, in_precs[i], getInputShapeAtPort(i), false, -1);
    }

    std::vector<PortConfigurator> out_configs;
    out_configs.reserve(getOriginalOutputsNumber());
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        out_configs.emplace_back(LayoutType::ncsp, out_precs[i], getOutputShapeAtPort(i), false, -1);
    }

    addSupportedPrimDesc(in_configs, out_configs, impl_desc_type::ref_any);
}

void PaKVReorder::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto& key_dims = getSrcMemoryAtPort(0)->getStaticDims();
    const auto& value_dims = getSrcMemoryAtPort(1)->getStaticDims();

    OPENVINO_ASSERT(key_dims.size() == 4, "PaKVReorder expects 4D key_cache");
    OPENVINO_ASSERT(value_dims.size() == 4, "PaKVReorder expects 4D value_cache");
    OPENVINO_ASSERT(key_dims[0] == value_dims[0] && key_dims[1] == value_dims[1], "PaKVReorder key/value cache mismatch");

    const size_t block_size = value_dims[2];
    OPENVINO_ASSERT(block_size > 0, "PaKVReorder expects positive block size");

    const size_t max_blocks = key_dims[0];
    const size_t kv_heads = key_dims[1];
    const size_t key_hidden = key_dims[2];
    const size_t key_block = key_dims[3];
    const size_t value_hidden = value_dims[3];

    const auto key_elem_size = getSrcMemoryAtPort(0)->getDesc().getPrecision().size();
    const auto value_elem_size = getSrcMemoryAtPort(1)->getDesc().getPrecision().size();

    auto* key_cache = getSrcDataAtPortAs<uint8_t>(0);
    auto* value_cache = getSrcDataAtPortAs<uint8_t>(1);
    const auto* block_indices = getSrcDataAtPortAs<int32_t>(2);
    const auto* block_indices_begins = getSrcDataAtPortAs<int32_t>(3);
    const auto* block_update_indices = getSrcDataAtPortAs<int32_t>(4);
    const auto* block_update_indices_begins = getSrcDataAtPortAs<int32_t>(5);

    const auto& block_indices_begins_dims = getSrcMemoryAtPort(3)->getStaticDims();
    const auto& block_update_indices_dims = getSrcMemoryAtPort(4)->getStaticDims();
    OPENVINO_ASSERT(!block_indices_begins_dims.empty(), "PaKVReorder expects non-empty block_indices_begins");

    const size_t seq_count = block_indices_begins_dims[0] - 1;
    const size_t block_update_values = block_update_indices_dims.empty() ? 0 : block_update_indices_dims[0];

    context->getCpuParallel()->parallel_for2d(seq_count, kv_heads, [&](size_t seq_idx, size_t head_idx) {
        const auto block_indices_base = static_cast<int64_t>(block_indices_begins[seq_idx]);
        const auto block_indices_end = static_cast<int64_t>(block_indices_begins[seq_idx + 1]);
        if (block_indices_base < 0 || block_indices_end < block_indices_base) {
            return;
        }

        const auto blocks_in_seq = static_cast<size_t>(block_indices_end - block_indices_base);
        const auto pos_begin = static_cast<int64_t>(block_update_indices_begins[seq_idx]);
        const auto pos_end = static_cast<int64_t>(block_update_indices_begins[seq_idx + 1]);

        for (int64_t pos = pos_begin; pos < pos_end; pos++) {
            if (pos < 0) {
                continue;
            }

            const auto pair_base = static_cast<size_t>(pos) * 2;
            if (pair_base + 1 >= block_update_values) {
                break;
            }

            const auto src_i = block_update_indices[pair_base + 0];
            const auto dst_i = block_update_indices[pair_base + 1];
            if (src_i < 0 || dst_i < 0) {
                continue;
            }

            const size_t src_local = static_cast<size_t>(src_i);
            const size_t dst_local = static_cast<size_t>(dst_i);
            const size_t src_block_in_seq = src_local / block_size;
            const size_t dst_block_in_seq = dst_local / block_size;

            if (src_block_in_seq >= blocks_in_seq || dst_block_in_seq >= blocks_in_seq) {
                continue;
            }

            const size_t src_slot = src_local % block_size;
            const size_t dst_slot = dst_local % block_size;

            const int32_t src_block_id = block_indices[block_indices_base + static_cast<int64_t>(src_block_in_seq)];
            const int32_t dst_block_id = block_indices[block_indices_base + static_cast<int64_t>(dst_block_in_seq)];
            if (src_block_id < 0 || dst_block_id < 0) {
                continue;
            }

            const size_t src_block = static_cast<size_t>(src_block_id);
            const size_t dst_block = static_cast<size_t>(dst_block_id);
            if (src_block >= max_blocks || dst_block >= max_blocks) {
                continue;
            }

            for (size_t k = 0; k < key_hidden; k++) {
                const size_t src_elem_off = ((src_block * kv_heads + head_idx) * key_hidden + k) * key_block + src_slot;
                const size_t dst_elem_off = ((dst_block * kv_heads + head_idx) * key_hidden + k) * key_block + dst_slot;
                cpu_memcpy(key_cache + dst_elem_off * key_elem_size, key_cache + src_elem_off * key_elem_size, key_elem_size);
            }

            const size_t src_value_elem_off = ((src_block * kv_heads + head_idx) * block_size + src_slot) * value_hidden;
            const size_t dst_value_elem_off = ((dst_block * kv_heads + head_idx) * block_size + dst_slot) * value_hidden;
            cpu_memcpy(value_cache + dst_value_elem_off * value_elem_size,
                       value_cache + src_value_elem_off * value_elem_size,
                       value_hidden * value_elem_size);
        }
    });

    getDstDataAtPortAs<uint8_t>(0)[0] = 0;
}

}  // namespace ov::intel_cpu::node
