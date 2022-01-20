﻿// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_fake_quantize.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::FuseFakeQuantizeTransformation, "FuseFakeQuantizeTransformation", 0);

FuseFakeQuantizeTransformation::FuseFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::FakeQuantize>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "FuseFakeQuantizeTransformation");
    this->register_matcher(m, callback);
}

bool FuseFakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<opset1::FakeQuantize> fakeQuantize = ov::as_type_ptr<ngraph::opset1::FakeQuantize>(m.get_match_root());
    do {
        fakeQuantize = handle(context, fakeQuantize);
    } while (fakeQuantize != nullptr);
    return true;
}

namespace fuse_fq {
namespace {

std::shared_ptr<Node> updateShape(std::shared_ptr<Node> op, const PartialShape& targetPShape) {
    assert(targetPShape.is_static());
    assert(op->get_output_partial_shape(0).is_static());
    const Shape targetShape = targetPShape.to_shape();
    const Shape shape = op->get_output_shape(0);

    if ((shape.size() < targetShape.size()) && (shape.size() > 1ul)) {
        op = fold<opset1::Unsqueeze>(
            op,
            std::make_shared<opset1::Constant>(ngraph::element::i32, Shape{ 1 }, std::vector<size_t>({ 0ul })));
    }
    return op;
}

std::shared_ptr<Node> getDataNode(const std::shared_ptr<Node>& eltwise) {
    if (!ov::is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(0))) {
        return eltwise->get_input_node_shared_ptr(0);
    }

    if (!ov::is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(1))) {
        return eltwise->get_input_node_shared_ptr(1);
    }

    return nullptr;
}

std::shared_ptr<opset1::Constant> getConstant(const std::shared_ptr<Node>& eltwise) {
    if (eltwise->get_input_size() != 2) {
        return nullptr;
    }

    std::shared_ptr<opset1::Constant> constant = ov::as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(1));
    if (constant != nullptr) {
        return constant;
    }

    return ov::as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(0));
}

bool eltwiseWithConstant(const std::shared_ptr<Node>& eltwise) {
    std::shared_ptr<opset1::Constant> constant = getConstant(eltwise);
    if (constant == nullptr) {
        return false;
    }

    Shape shape = constant->get_shape();
    if ((!shape.empty()) && (shape_size(shape) != 1ul)) {
        const auto eltwisePShape = eltwise->get_output_partial_shape(0);
        if (eltwisePShape.rank().is_dynamic()) {
            return false;
        }

        const size_t eltwiseOutRank = eltwisePShape.rank().get_length();
        if ((eltwiseOutRank - shape.size()) > 1) {
            return false;
        }

        if ((eltwiseOutRank - shape.size()) == 1ul) {
            shape.insert(shape.begin(), 1ul);
        }

        for (size_t i = 2ul; i < shape.size(); ++i) {
            if (shape[i] != 1ul) {
                return false;
            }
        }
    }

    return getDataNode(eltwise) != nullptr;
}

}  // namespace
}  // namespace fuse_fq

std::shared_ptr<opset1::FakeQuantize> FuseFakeQuantizeTransformation::handle(
    TransformationContext& context,
    const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const {
    const std::shared_ptr<Node> eltwise = fakeQuantize->get_input_node_shared_ptr(0);

    std::shared_ptr<Node> inputLowConst = fakeQuantize->get_input_node_shared_ptr(1);
    std::shared_ptr<Node> inputHightConst = fakeQuantize->get_input_node_shared_ptr(2);

    std::shared_ptr<opset1::Constant> constant = fuse_fq::getConstant(eltwise);
    if (ov::is_type<opset1::Multiply>(eltwise) && fuse_fq::eltwiseWithConstant(eltwise)) {
        const auto value = constant->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            constant :
            foldConvert(constant, eltwise->get_output_element_type(0));

        inputLowConst = fuse_fq::updateShape(fold<opset1::Divide>(inputLowConst, value), fakeQuantize->get_output_partial_shape(0));
        inputHightConst = fuse_fq::updateShape(fold<opset1::Divide>(inputHightConst, value), fakeQuantize->get_output_partial_shape(0));
    } else if (ov::is_type<opset1::Divide>(eltwise) && fuse_fq::eltwiseWithConstant(eltwise)) {
        const auto value = constant->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            constant :
            foldConvert(constant, eltwise->get_output_element_type(0));

        inputLowConst = fuse_fq::updateShape(fold<opset1::Multiply>(inputLowConst, value), fakeQuantize->get_output_partial_shape(0));
        inputHightConst = fuse_fq::updateShape(fold<opset1::Multiply>(inputHightConst, value), fakeQuantize->get_output_partial_shape(0));
    } else if (ov::is_type<opset1::Subtract>(eltwise) && fuse_fq::eltwiseWithConstant(eltwise)) {
        const auto value = constant->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            constant :
            foldConvert(constant, eltwise->get_output_element_type(0));

        inputLowConst = fuse_fq::updateShape(fold<opset1::Add>(inputLowConst, value), fakeQuantize->get_output_partial_shape(0));
        inputHightConst = fuse_fq::updateShape(fold<opset1::Add>(inputHightConst, value), fakeQuantize->get_output_partial_shape(0));
    } else if (ov::is_type<opset1::Add>(eltwise) && fuse_fq::eltwiseWithConstant(eltwise)) {
        if (ov::is_type<opset1::Convolution>(fuse_fq::getDataNode(eltwise)) ||
            ov::is_type<opset1::GroupConvolution>(fuse_fq::getDataNode(eltwise))) {
            return nullptr;
        }

        const auto value = constant->get_output_element_type(0) == eltwise->get_output_element_type(0) ?
            constant :
            foldConvert(constant, eltwise->get_output_element_type(0));

        inputLowConst = fuse_fq::updateShape(fold<opset1::Subtract>(inputLowConst, value), fakeQuantize->get_output_partial_shape(0));
        inputHightConst = fuse_fq::updateShape(fold<opset1::Subtract>(inputHightConst, value), fakeQuantize->get_output_partial_shape(0));
    } else if (ov::is_type<opset1::Convert>(eltwise)) {
        // issue #40611
        if ((eltwise->get_input_element_type(0) == element::i32) && (eltwise->get_output_element_type(0) == element::f32)) {
            return nullptr;
        }
    } else {
        return nullptr;
    }

    const auto data = fuse_fq::getDataNode(eltwise);
    const size_t outputIdx = NetworkHelper::getParentOutputIndex(data, eltwise);

    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = ov::as_type_ptr<opset1::FakeQuantize>(fakeQuantize->clone_with_new_inputs({
        data->output(outputIdx),
        inputLowConst,
        inputHightConst,
        fakeQuantize->input_value(3),
        fakeQuantize->input_value(4) }));

    replace_node(fakeQuantize, newFakeQuantize);
    NetworkHelper::copyInfo(fakeQuantize, newFakeQuantize);
    return newFakeQuantize;
}

bool FuseFakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
