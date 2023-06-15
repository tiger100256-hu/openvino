// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/interpolate.hpp>

#include "utils.hpp"

namespace ov {
namespace op {

namespace util {

template <typename T>
void correct_pads_attr(const util::InterpolateBase* op,
                       std::vector<size_t>& pads_begin,
                       std::vector<size_t>& pads_end,
                       const std::vector<T>& input_shapes) {
    auto input_shape = input_shapes[0];
    if (input_shape.rank().is_dynamic()) {
        return;
    }
    const auto input_rank = input_shape.size();

    pads_begin = op->m_attrs.pads_begin;
    pads_end = op->m_attrs.pads_end;
    if (pads_begin.size() != input_rank) {
        pads_begin.resize(input_rank);
    }
    if (pads_end.size() != input_rank) {
        pads_end.resize(input_rank);
    }
}

inline int64_t multiply_bound_and_scale(int64_t bound, float scale) {
    if (bound == -1) {
        return bound;
    }
    return static_cast<int64_t>(static_cast<float>(bound) * scale);
}

template <typename T>
void infer_using_scales(T& output_shape, const std::vector<int64_t>& axes, const std::vector<float>& scales) {
    size_t i = 0;
    static constexpr float epsilon = 1.0e-6f;
    for (const auto& axis : axes) {
        if (scales[i] == 1.) {
            ++i;
            continue;
        }
        const auto& current_dim = output_shape[axis];
        float multiplier = scales[i] + epsilon;
        if (current_dim.is_static()) {
            output_shape[axis] = multiply_bound_and_scale(current_dim.get_length(), multiplier);
        } else {
            int64_t new_lower_bound = multiply_bound_and_scale(current_dim.get_min_length(), multiplier);
            int64_t new_upper_bound = multiply_bound_and_scale(current_dim.get_max_length(), multiplier);
            output_shape[axis] = ov::Dimension(new_lower_bound, new_upper_bound);
        }
        ++i;
    }
}
}  // namespace util

namespace v0 {
template <class T>
void shape_infer(const Interpolate* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    output_shape = input_shape;
    const auto& attr = op->get_attrs();

    if (input_shape.rank().is_static()) {
        auto input_rank = input_shape.size();
        NODE_VALIDATION_CHECK(op,
                              std::all_of(attr.axes.begin(),
                                          attr.axes.end(),
                                          [input_rank](size_t axis) {
                                              return axis < input_rank;
                                          }),
                              "Axis value should less than input rank. ",
                              "Got: input rank ",
                              input_rank,
                              ", axes ",
                              attr.axes);

        T target_spatial_shape;
        if (get_data_as_shape<T>(1, op, target_spatial_shape, constant_data)) {
            size_t i = 0;
            for (auto axis : attr.axes) {
                output_shape[axis] = target_spatial_shape[i++];
            }
        } else {
            for (auto axis : attr.axes) {
                output_shape[axis] = ov::Dimension::dynamic();
            }
        }
    }
}
}  // namespace v0

namespace v4 {
template <class T>
void shape_infer(const Interpolate* op,
                 std::vector<size_t>& pads_begin,
                 std::vector<size_t>& pads_end,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 4) && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    const auto& input_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    output_shape = input_shape;

    if (input_shape.rank().is_static()) {
        const auto input_rank = input_shape.size();

        // Get axes
        std::vector<int64_t> axes;
        if (input_shapes.size() == 4 && !(get_data_as_int64<T>(3, op, axes, constant_data))) {
            for (size_t i = 0; i < input_rank; i++)
                output_shape[i] = ov::Dimension::dynamic();
            return;
        } else if (input_shapes.size() == 3) {
            axes.resize(input_rank);
            std::iota(axes.begin(), axes.end(), 0);
        }
        NODE_VALIDATION_CHECK(op,
                              std::all_of(axes.begin(),
                                          axes.end(),
                                          [input_rank](size_t axis) {
                                              return axis < input_rank;
                                          }),
                              "Axis value should less than input rank.");

        // Get padded input shape
        for (size_t i = 0; i < input_rank; ++i) {
            output_shape[i] = DimType(pads_begin[i]) + DimType(pads_end[i]) + input_shape[i];
        }

        if (op->m_attrs.shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES) {
            std::vector<float> scales;
            if (get_data_as_float<T>(2, op, scales, constant_data)) {
                util::infer_using_scales(output_shape, axes, scales);
            } else {
                for (const auto& axis : axes) {
                    output_shape[axis] = ov::Dimension::dynamic();
                }
            }
        } else {
            T target_spatial_shape;
            if (get_data_as_shape<T>(1, op, target_spatial_shape, constant_data)) {
                size_t i = 0;
                for (const auto& axis : axes) {
                    output_shape[axis] = target_spatial_shape[i++];
                }
            } else {
                for (const auto& axis : axes) {
                    output_shape[axis] = ov::Dimension::dynamic();
                }
            }
        }
    }
}

}  // namespace v4

namespace v11 {
template <class T>
void shape_infer(const Interpolate* op,
                 std::vector<size_t>& pads_begin,
                 std::vector<size_t>& pads_end,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3) && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    const auto& input_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    output_shape = input_shape;

    if (input_shape.rank().is_static()) {
        const auto input_rank = input_shape.size();

        // Get axes
        std::vector<int64_t> axes;
        if (input_shapes.size() == 3 && !(get_data_as_int64<T>(2, op, axes, constant_data))) {
            for (size_t i = 0; i < input_rank; i++)
                output_shape[i] = ov::Dimension::dynamic();
            return;
        } else if (input_shapes.size() == 2) {
            axes.resize(input_rank);
            std::iota(axes.begin(), axes.end(), 0);
        }
        NODE_VALIDATION_CHECK(op,
                              std::all_of(axes.begin(),
                                          axes.end(),
                                          [input_rank](size_t axis) {
                                              return axis < input_rank;
                                          }),
                              "Axis value should be smaller than the input's rank.");

        // Get padded input shape
        for (size_t i = 0; i < input_rank; ++i) {
            output_shape[i] = DimType(pads_begin[i]) + DimType(pads_end[i]) + input_shape[i];
        }

        if (op->get_attrs().shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES) {
            std::vector<float> scales;
            if (get_data_as_float<T>(1, op, scales, constant_data)) {
                NODE_VALIDATION_CHECK(
                    op,
                    axes.size() == scales.size(),
                    "The number of elements in the 'scales_or_sizes' input does not match the number of axes");
                util::infer_using_scales(output_shape, axes, scales);
            } else {
                for (const auto& axis : axes) {
                    output_shape[axis] = ov::Dimension::dynamic();
                }
            }
        } else {
            T target_spatial_shape;
            if (get_data_as_shape<T>(1, op, target_spatial_shape, constant_data)) {
                NODE_VALIDATION_CHECK(
                    op,
                    axes.size() == target_spatial_shape.size(),
                    "The number of elements in the 'scales_or_sizes' input does not match the number of axes");
                size_t i = 0;
                for (const auto& axis : axes) {
                    output_shape[axis] = target_spatial_shape[i++];
                }
            } else {
                for (const auto& axis : axes) {
                    output_shape[axis] = ov::Dimension::dynamic();
                }
            }
        }
    }
}
}  // namespace v11
}  // namespace op
}  // namespace ov
