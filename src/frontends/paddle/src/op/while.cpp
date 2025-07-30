// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/while.hpp"

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include <cassert>

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

using namespace default_opset;

NamedOutputs while_(const NodeContext& node) {
    const auto data = node.get_ng_inputs("X");
    const auto cond = node.get_input("Condition");
    int32_t sub_block = -1;
    if (node.is_json_format()) {
        auto sub_block_index = node.get_input("sub_block_index");
        auto sub_block_index_node = sub_block_index.get_node_shared_ptr();
        auto sub_block_index_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(sub_block_index_node);
        auto sub_block_index_value_vector = sub_block_index_const->cast_vector<int32_t>();
        assert(sub_block_index_value_vector.size() == 1);
        sub_block = sub_block_index_value_vector[0];
    } else {
        sub_block = node.get_attribute<int32_t>("sub_block");
    }
    auto outputs_info = node.get_output_port_infos("Out");

    ov::OutputVector inputs = data;
    inputs.push_back(cond);
    NamedOutputs named_outputs;
    named_outputs["Out"] = std::make_shared<ov::op::internal::While>(inputs, sub_block, outputs_info,
                                                                     node.is_json_format())->outputs();
    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
