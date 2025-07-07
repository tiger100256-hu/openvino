// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_if.hpp"

#include "default_opset.hpp"
#include "internal/op/conditional_block.hpp"
#include "internal/op/tensorarray_write.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/fold_subgraph_empty_inputs.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace ov::frontend::paddle::op::default_opset;

// Transform Paddle "if" to OpenVINO If op.
ov::frontend::paddle::pass::TransformIfElse::TransformIfElse(std::vector<std::shared_ptr<Model>> funcs) {
    const auto cond_label = pattern::wrap_type<ov::op::internal::IfElseBlock>();

    matcher_pass_callback callback = [funcs](pattern::Matcher& m) -> bool {
        const auto if_else_block = ov::as_type_ptr<ov::op::internal::IfElseBlock>(m.get_match_root());
        if (!if_else_block) {
            return false;
        }
        const auto mask_idx = if_else_block->get_input_size() - 1;
        const auto cond = if_else_block->get_input_node_shared_ptr(mask_idx);

        if (!cond) {
            return false;
        }

        // build_if_node
        const auto if_else_block_ids = if_else_block->get_subblock_indexs();
        OPENVINO_ASSERT(if_else_block_ids.size() == 2, "there should be two branch here);
        const auto then_idx = if_else_block_ids[0]
        const auto& then_branch = funcs[then_idx];
        const auto& then_params = then_branch->get_parameters();
        const auto else_idx = if_else_block_ids[1]
        const auto& else_branch = funcs[else_idx];
        const auto& else_params = else_branch->get_parameters();

        auto if_node = std::make_shared<If>(cond);
        ov::pass::disable_fold_subgraph_empty_inputs(if_node);
        if_node->set_then_body(then_branch);
        if_node->set_else_body(else_branch);

        // get inputs
        const auto then_branch_inputs_from_parent = if_else_block->get_inputs_from_parent(0);
        auto then_param = then_params.cbegin();
        for (const auto& from_parent : then_branch_inputs_from_parent) {
            if_node->set_input(from_parent, *then_param, nullptr);
            then_param++;
        }
        const auto else_branch_inputs_from_parent = if_else_block->get_inputs_from_parent(1);
        auto else_param = else_params.cbegin();
        for (const auto& from_parent : else_branch_inputs_from_parent) {
            if_node->set_input(from_parent,  nullptr, *else_param);
            else_param++;
        }

        auto then_results = then_branch->get_results();
        auto else_results = else_branch->get_results();
        for (size_t i = 0; i < else_results.size(); i++) {
            if_node->set_output(then_results[i], else_results[i]);
        }
        replace_node(conditional_block, if_node);
        if_node->set_friendly_name(conditional_block->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(cond_label, "if_else_block");
    this->register_matcher(m, callback);
}
