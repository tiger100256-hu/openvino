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
        const auto then_inputs = node.get_ng_input("then_inputs");
        const auto then_params = node.get_ng_input("then_params");
        const auto then_outputs = node.get_ng_input("then_outputs");
        const auto else_inputs = node.get_ng_input("else_inputs");
        const auto else_pararms = node.get_ng_input("else_params");
        const auto else_outputs = node.get_ng_input("else_outputs");
        const auto if_node = std::make_shared<default_opset::If>(cond);
        const auto then_branch = std::make_shared<Model>(then_outputs, then_inputs);
        const auto else_branch = std::make_shared<Model>(else_outputs, then_inputs);
        if_node->set_then_body(then_branch);
        if_node->set_else_body(else_branch);
        for (size_t i = 0; i < then_params.size(); i++) {
            if_node->set_input(then_inputs[i], then_params[i], nullptr);
        }
        for (size_t i = 0; i < else_params.size(); i++) {
            if_node->set_input(else_inputs[i], nullptr, else_params[i]);
        }
        auto else_results = else_branch->get_results();
        auto then_results = then_branch->get_results();
        for (size_t i = 0; i < else_results.size(); i++) {
            if_node->set_output(then_results[i], else_results[i]);
        }
        return node.default_single_output_mapping({if_node}, {"Out"});
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
