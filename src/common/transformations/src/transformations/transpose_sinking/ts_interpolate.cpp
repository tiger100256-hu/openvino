// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_interpolate.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

TSInterpolateForward::TSInterpolateForward() {
    MATCHER_SCOPE(TSInterpolateForward);
    auto const_label = wrap_type<ov::op::v0::Constant>();
    auto transpose_label = wrap_type<ov::op::v1::Transpose>({any_input(), const_label});
    auto main_node_label = wrap_type<ov::op::v4::Interpolate>({transpose_label, any_input(), any_input(), any_input()});

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_node = m.get_pattern_map();

        auto& main_node = pattern_to_node.at(main_node_label);
        if (transformation_callback(main_node)) {
            return false;
        }

        auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(pattern_to_node.at(transpose_label));
        if (!transpose) {
            return false;
        }

        auto transpose_const = as_type_ptr<ov::op::v0::Constant>(pattern_to_node.at(const_label));
        if (!transpose_const) {
            return false;
        }

        // remove Transpose on 1st input:
        auto transpose_parent = transpose->input_value(0);
        main_node->input(0).replace_source_output(transpose_parent);

        const auto transpose_axis_order = transpose_const->get_axis_vector_val();
        auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);

        const auto& interpolate = std::dynamic_pointer_cast<ov::op::v4::Interpolate>(main_node);
        const auto& new_axes = ChangeAxes(main_node->input_value(3), transpose_axis_order, axis);
        main_node->input(3).replace_source_output(new_axes);

        if (interpolate) {
            op::v4::Interpolate::InterpolateAttrs attrs = interpolate->get_attrs();
            if (!attrs.pads_begin.empty() || !attrs.pads_end.empty()) {
                const auto& order_size = transpose_axis_order.size();
                attrs.pads_begin.resize(order_size);
                attrs.pads_end.resize(order_size);
                std::vector<size_t> new_pads_begin(order_size), new_pads_end(order_size);
                for (size_t i = 0; i < order_size; ++i) {
                    new_pads_begin[i] = attrs.pads_begin[transpose_axis_order[i]];
                    new_pads_end[i] = attrs.pads_end[transpose_axis_order[i]];
                }
                std::swap(attrs.pads_begin, new_pads_begin);
                std::swap(attrs.pads_end, new_pads_end);
                interpolate->set_attrs(attrs);
            }
        }

        main_node->validate_and_infer_types();
        TransposeInputsInfo transpose_input_info = {transpose, transpose_const, 0};
        for (auto& new_node : sink_forward::InsertOutputTransposes(main_node, transpose_input_info)) {
            register_new_node(new_node);
            UpdateForwardSinkingAbility(new_node);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(main_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

TSInterpolateBackward::TSInterpolateBackward() {
    MATCHER_SCOPE(TSInterpolateBackward);

    auto main_node_label = wrap_type<ov::op::v4::Interpolate>([](const Output<Node>& output) -> bool {
        return has_static_rank()(output) && CheckTransposeConsumers(output);
    });

    auto transpose_const_label = wrap_type<ov::op::v0::Constant>();

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({main_node_label, transpose_const_label},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
            as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();
        if (transformation_callback(main_node)) {
            return false;
        }

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node,
                                                                       transpose_const,
                                                                       /* input_indexes= */ {0})) {
            register_new_node(new_node);
        }

        RemoveTransposeConsumers(main_node);
        const auto transpose_axis_order = transpose_const->get_axis_vector_val();
        const auto reversed_transpose_order = ReverseTransposeOrder(transpose_axis_order);
        auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
        auto new_axes = ChangeAxes(main_node->input_value(3), reversed_transpose_order, axis);
        main_node->input(3).replace_source_output(new_axes);

        const auto& interpolate = std::dynamic_pointer_cast<ov::op::v4::Interpolate>(main_node);
        if (interpolate) {
            op::v4::Interpolate::InterpolateAttrs attrs = interpolate->get_attrs();
            if (!attrs.pads_begin.empty() || !attrs.pads_end.empty()) {
                const auto& order_size = reversed_transpose_order.size();
                attrs.pads_begin.resize(order_size);
                attrs.pads_end.resize(order_size);
                std::vector<size_t> new_pads_begin(order_size), new_pads_end(order_size);
                for (size_t i = 0; i < order_size; ++i) {
                    new_pads_begin[i] = attrs.pads_begin[reversed_transpose_order[i]];
                    new_pads_end[i] = attrs.pads_end[reversed_transpose_order[i]];
                }
                std::swap(attrs.pads_begin, new_pads_begin);
                std::swap(attrs.pads_end, new_pads_end);
                interpolate->set_attrs(attrs);
            }
        }
        main_node->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
