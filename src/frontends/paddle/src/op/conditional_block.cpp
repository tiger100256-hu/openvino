// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/conditional_block.hpp"

#include "default_opset.hpp"
#include "internal/op/while.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs conditional_block(const NodeContext& node) {
    if (node.is_json_format()) {
        const auto cond = node.get_input("cond");
        const auto block_ids = node.get_input("block_ids");
        auto block_ids_node = full.get_node_shared_ptr();
        auto block_ids_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(block_ids_node);
        auto block_ids_value_vector = block_ids_const->cast_vector<int32_t>();
        assert(block_ids_value_vector.size() == 2);
        block_ids = block_ids_value_vector[0];
        std::shared_ptr<Node> placehodler_if;
        std::shared_ptr<Node> placehodler_else;
        const auto outputs_info = node.get_output_port_infos("Out");
        placehodler_if = std::make_shared<ov::op::internal::ConditionalBlock>(cond, true, block_ids[0], outputs_info);
        Output<Node> no_cond = std::make_shared<ov::opset6::LogicalNot>(cond)
        placehodler_else = std::make_shared<ov::op::internal::ConditionalBlock>(no_cond, true, block_ids[1], outputs_info);
    }
    const auto cond = node.get_input("Cond");
    const auto sub_block = node.get_attribute<int32_t>("sub_block");
    const auto is_scalar_condition = node.get_attribute<bool>("is_scalar_condition", true);

    const auto outputs_info = node.get_output_port_infos("Out");

    std::shared_ptr<Node> placehodler;
    if (node.has_input("Input")) {
        const auto inputs = node.get_ng_inputs("Input");
        placehodler = std::make_shared<ov::op::internal::ConditionalBlock>(inputs,
                                                                           cond,
                                                                           is_scalar_condition,
                                                                           sub_block,
                                                                           outputs_info);
    } else {
        placehodler =
            std::make_shared<ov::op::internal::ConditionalBlock>(cond, is_scalar_condition, sub_block, outputs_info);
    }
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
