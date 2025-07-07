// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/conditional_block.hpp"

#include <algorithm>

#include "openvino/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

op::internal::IfElseBlock::IfElseBlock(
    const Output<Node>& cond,
    const OutputVector& if_inputs,
    const OutputVector& else_inputs,
    std::vector<int32_t> sub_block_indexs,
    const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos)
    : Op({cond}),
      m_sub_block_indexs(sub_block_indexs),
      m_output_infos(output_infos) {
      m_inputs_from_parent.push_back(if_inputs);
      m_inputs_from_parent.push_back(else_inputs);
    constructor_validate_and_infer_types();
}

// op::internal::IfElseBlock::IfElseBlock(
//     const OutputVector& inputs,
//     const Output<Node>& cond,
//     bool is_scalar_condition,
//     int32_t sub_block_index,
//     const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos)
//     : m_is_scalar_condition(is_scalar_condition),
//       m_sub_block_index(sub_block_index),
//       m_output_infos(output_infos) {
//     OutputVector new_args;
//     std::move(inputs.begin(), inputs.end(), std::back_inserter(new_args));
//     new_args.emplace_back(cond);
//     set_arguments(new_args);
//     constructor_validate_and_infer_types();
// }

std::shared_ptr<Node> op::internal::IfElseBlock::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<IfElseBlock>(new_args.at(0), m_inputs_from_parent[0], m_inputs_from_parent[1], m_sub_block_indexs, m_output_infos);
}

bool op::internal::IfElseBlock::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("sub_block_indexs", m_sub_block_indexs);
    return true;
}

void op::internal::IfElseBlock::validate_and_infer_types() {
    for (size_t i = 0; i < m_output_infos.size(); i++) {
        set_output_type(i, m_output_infos[i].first, m_output_infos[i].second);
    }
}

const std::vecotr<OutputVector>& op::internal::IfElseBlock::get_inputs_from_parent(size_t index) const {
    assert(index < m_inputs_from_parent.size());
    return m_inputs_from_parent[index];
}
