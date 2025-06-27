// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"
#include <cassert>

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs clip(const NodeContext& node) {
    auto data = node.get_input("X");
    float min;
    float max;
    if (node.is_json_format()) {
        auto min_input = node.get_input("min");
        auto min_node = min_input.get_node_shared_ptr();
        auto min_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(min_node);
        auto min_value_vector = min_const->get_vector<float>();
        assert(min_value_vector.size() == 1);
        min = min_value_vector[0];
        auto max_input = node.get_input("max");
        auto max_node = max_input.get_node_shared_ptr();
        auto max_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(max_node);
        auto max_value_vector = max_const->get_vector<float>();
        assert(max_value_vector.size() == 1);
        max = max_value_vector[0];
    } else {
        min = node.get_attribute<float>("min");
        max = node.get_attribute<float>("max");
    }
    PADDLE_OP_CHECK(node, max >= min, "clip: max value must greater than min value!");

    return node.default_single_output_mapping({std::make_shared<ov::opset6::Clamp>(data, min, max)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
