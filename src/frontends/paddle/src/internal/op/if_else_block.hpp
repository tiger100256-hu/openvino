// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class IfElseBlock : public Op {
public:
    OPENVINO_OP("IfElseBlock", "internal");

    IfElseBlock() = default;

    // IfElseBlock(const OutputVector& inputs,
    //                  const Output<Node>& cond,
    //                  std::vector<int32_t>& sub_block_indexs,
    //                  const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos);
    IfElseBlock(const Output<Node>& cond,
                const OutputVector& if_inputs,
                const OutputVector& else_inputs,
                const std::vector<int32_t>& sub_block_indexs,
                const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return A vector containing the values for each input except "cond".
    const OutputVector& get_inputs_from_parent(size_t index) const;

    const std::vector<int32_t>& get_subblock_indexs() const {
        return m_sub_block_indexs;
    }

private:
    std::vector<int32_t> m_sub_block_indexs;
    std::vector<OutputVector> m_inputs_from_parent;
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> m_output_infos;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
