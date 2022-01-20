// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/binary_elementwise_comparison.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/elementwise_args.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::BinaryElementwiseComparison);

ov::op::util::BinaryElementwiseComparison::BinaryElementwiseComparison(const AutoBroadcastSpec& autob)
    : m_autob(autob) {}

ov::op::util::BinaryElementwiseComparison::BinaryElementwiseComparison(const Output<Node>& arg0,
                                                                       const Output<Node>& arg1,
                                                                       const AutoBroadcastSpec& autob)
    : Op({arg0, arg1}),
      m_autob(autob) {}

void ov::op::util::BinaryElementwiseComparison::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v0_util_BinaryElementwiseComparison_validate_and_infer_types);
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this, m_autob);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    set_output_type(0, element::boolean, args_pshape);
}

bool ov::op::util::BinaryElementwiseComparison::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v0_util_BinaryElementwiseComparison_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}
