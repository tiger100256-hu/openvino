// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <limits.h>

#include "default_opset.hpp"
#include "op_utils.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
namespace {
Output<Node> idx_node(const std::string& tensor_alias,
                      const std::string& list_alias,
                      const std::string& attr_alias,
                      const NodeContext& node) {
    if (node.has_input(tensor_alias)) {
        return std::make_shared<default_opset::Convert>(node.get_input(tensor_alias), element::i32);
    } else if (node.has_input(list_alias)) {
        auto inputs = node.get_ng_inputs(list_alias);
        return std::make_shared<default_opset::Convert>(get_tensor_list(inputs), element::i32);
    } else {
        auto values = node.get_attribute<std::vector<int32_t>>(attr_alias);
        return default_opset::Constant::create(element::i32, {values.size()}, values);
    }
}
NamedOutputs slice_op(const NodeContext& node, const bool& stride_input) {
    const auto data = node.get_input("Input");
    std::vector<int32_t> axes;
    if (node.is_json_format()) {
        auto axes_any = node.get_attribute_as_any("axes");
        if (axes_any.is<std::vector<int64_t>>()) {
            auto axes_64 = axes_any.as<std::vector<int64_t>>();
            axes.resize(axes_64.size());
            std::transform(axes_64.begin(), axes_64.end(), axes.begin(), [](int64_t value) {
                    return static_cast<int32_t>(value);
                    });
        } else if (axes_any.is<std::vector<int32_t>>()) {
            axes = axes_any.as<std::vector<int32_t>>();
        } else {
            PADDLE_OP_CHECK(node,
                    false,
                    "axes format is not i32 or i64.");
        }
    } else {
        axes = node.get_attribute<std::vector<int32_t>>("axes");
    }

    Output<Node> start_idx_node = idx_node("StartsTensor", "StartsTensorList", "starts", node);
    Output<Node> end_idx_node = idx_node("EndsTensor", "EndsTensorList", "ends", node);
    Output<Node> strides_idx_node;
    if (stride_input) {
        strides_idx_node = idx_node("StridesTensor", "StridesTensorList", "strides", node);
    } else {
        strides_idx_node =
            default_opset::Constant::create(element::i32, start_idx_node.get_shape(), std::vector<int32_t>{1});
    }
    const auto axes_node = default_opset::Constant::create(element::i32, {axes.size()}, axes);
    const auto slice_node =
        std::make_shared<default_opset::Slice>(data, start_idx_node, end_idx_node, strides_idx_node, axes_node);
    std::vector<int32_t> decrease_axis;
    if (node.is_json_format()) {
        auto decrease_axis_64 = node.get_attribute<std::vector<int64_t>>("decrease_axis", {});
        for (auto& item : decrease_axis_64) {
            decrease_axis.push_back(item);
        }
    } else {
        decrease_axis = node.get_attribute<std::vector<int32_t>>("decrease_axis");
    }

    if (decrease_axis.size() > 0) {
        PartialShape input_shape = data.get_partial_shape();
        PADDLE_OP_CHECK(node,
                        input_shape.rank().is_static(),
                        "input rank of slice must be static when decrease_axis is set.");
        if (input_shape.size() == decrease_axis.size()) {
            // according to paddle slice_op, when all axes are decreased, output shape is [1], instead of scalar.
            // Ref: paddle/fluid/operators/slice_op.h
            auto decreased_node = std::make_shared<default_opset::Reshape>(
                slice_node,
                std::make_shared<default_opset::Constant>(element::i64, Shape{1}, 1),
                false);
            const auto output_info = node.get_output_port_infos("Out");
            size_t output_size = output_info[0].second.size();
            if (output_size == 0) {
                auto squeeze_node = std::make_shared<default_opset::Squeeze>(decreased_node);
                return node.default_single_output_mapping({squeeze_node}, {"Out"});
            }
            return node.default_single_output_mapping({decreased_node}, {"Out"});
        }

        const auto squeeze_index_node =
            default_opset::Constant::create(element::i32, {decrease_axis.size()}, decrease_axis);
        const auto decreased_node = std::make_shared<default_opset::Squeeze>(slice_node, squeeze_index_node);
        return node.default_single_output_mapping({decreased_node}, {"Out"});
    } else {
        return node.default_single_output_mapping({slice_node}, {"Out"});
    }
}
}  // namespace
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
