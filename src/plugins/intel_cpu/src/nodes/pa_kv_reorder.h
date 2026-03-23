// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"

namespace ov::intel_cpu::node {

class PaKVReorder : public Node {
public:
    PaKVReorder(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;

    bool created() const override {
        return getType() == Type::PaKVReorder;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
};

}  // namespace ov::intel_cpu::node
