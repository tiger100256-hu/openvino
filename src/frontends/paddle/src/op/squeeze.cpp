// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs squeeze(const NodeContext& node) {
    auto data = node.get_input("X");
    std::vector<int32_t> axes;
    if (node.is_json_format()) {
        auto full = node.get_input("full");
        auto axis_node = full.get_node_shared_ptr();
        auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(axis_node);
        axes = axis_const->cast_vector<int32_t>();
    } else if (node.has_attribute("axes")) {
        axes = node.get_attribute<std::vector<int32_t>>("axes");
    }

    std::shared_ptr<Node> out;
    if (!axes.empty()) {
        auto axesNode = ov::opset6::Constant::create(ov::element::i32, {axes.size()}, axes);
        out = std::make_shared<ov::opset6::Squeeze>(data, axesNode);
    } else {
        out = std::make_shared<ov::opset6::Squeeze>(data);
    }
    return node.default_single_output_mapping(out, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
