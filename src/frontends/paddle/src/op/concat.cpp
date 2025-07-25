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
NamedOutputs concat(const NodeContext& node) {
    auto data = node.get_ng_inputs("X");
    if (node.is_json_format()) {
        auto full = node.get_input("full");
        auto axis_node = full.get_node_shared_ptr();
        auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(axis_node);
        auto axis_value_vector = axis_const->cast_vector<int32_t>();
        assert(axis_value_vector.size() == 1);
        int axis = axis_value_vector[0];
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Concat>(data, axis)}, {"Out"});
    } else {
        auto axis = node.get_attribute<int>("axis");
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Concat>(data, axis)}, {"Out"});
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
