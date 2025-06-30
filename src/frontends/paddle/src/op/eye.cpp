// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include <cassert>

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs eye(const NodeContext& node) {
    int64_t row;
    int64_t col;
    if (node.is_json_format()) {
        auto full = node.get_input("num_rows");
        auto rows_node = full.get_node_shared_ptr();
        auto rows_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(rows_node);
        auto row_vector = rows_const->cast_vector<int64_t>();
        assert(row_vector.size() == 1);
        row = row_vector[0];
        full = node.get_input("num_columns");
        auto cols_node = full.get_node_shared_ptr();
        auto cols_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(cols_node);
        auto col_vector = cols_const->cast_vector<int64_t>();
        assert(col_vector.size() == 1);
        col = col_vector[0];
    } else {
        row = node.get_attribute<int64_t>("num_rows");
        col = node.get_attribute<int64_t>("num_columns", row);
    }
    auto dtype = node.get_attribute<ov::element::Type>("dtype", ov::element::f32);

    const auto& row_node = std::make_shared<default_opset::Constant>(ov::element::i64, Shape{}, (row));
    const auto& col_node = std::make_shared<default_opset::Constant>(ov::element::i64, Shape{}, (col));
    const auto& diagonal_index_node = std::make_shared<default_opset::Constant>(ov::element::i32, Shape{}, (0));

    std::shared_ptr<Node> out_node;
    if (dtype == ov::element::i32 || dtype == ov::element::i64) {
        out_node = std::make_shared<default_opset::Eye>(row_node, col_node, diagonal_index_node, dtype);
    } else {
        const auto& eye_node =
            std::make_shared<default_opset::Eye>(row_node, col_node, diagonal_index_node, ov::element::i32);
        out_node = std::make_shared<default_opset::Convert>(eye_node, dtype);
    }

    return node.default_single_output_mapping({out_node}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
