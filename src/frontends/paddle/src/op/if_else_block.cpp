// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/if_else_block.hpp"

#include "default_opset.hpp"
#include "internal/op/while.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include <cassert>

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs if_else_block(const NodeContext& node) {
    const auto cond = node.get_input("Cond");
    const auto if_inputs = node.get_ng_inputs("if_inputs");
    const auto else_inputs = node.get_ng_inputs("else_inputs");
    const auto sub_block_index = node.get_input("sub_block_indexs");
    auto sub_block_index_node = sub_block_index.get_node_shared_ptr();
    auto sub_block_index_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(sub_block_index_node);
    auto sub_block_index_value_vector = sub_block_index_const->cast_vector<int32_t>();
    assert(sub_block_index_value_vector.size() == 2);
    const auto outputs_info = node.get_output_port_infos("Out");
    std::shared_ptr<Node> placehodler;
    placehodler = std::make_shared<ov::op::internal::IfElseBlock>(cond,
                  if_inputs, else_inputs,
                  sub_block_index_value_vector,
                  outputs_info);
    const auto outputs = placehodler->outputs();

    auto out_names = node.get_output_names();
    auto it = std::find(out_names.begin(), out_names.end(), "Out");
    PADDLE_OP_CHECK(node, it != out_names.end(), "Expected output not found");

    NamedOutputs named_outputs;
    for (const auto& output : outputs) {
        named_outputs[*it].push_back(output);
    }
    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
