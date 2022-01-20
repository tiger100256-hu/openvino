// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reduce_kernel_base.h"
#include <vector>

namespace kernel_selector {
class ReduceKernel_b_fs_yx_fsv16 : public ReduceKernelBase {
public:
    ReduceKernel_b_fs_yx_fsv16() : ReduceKernelBase("reduce_gpu_b_fs_yx_fsv16") {}
    virtual ~ReduceKernel_b_fs_yx_fsv16() {}
    CommonDispatchData SetDefault(const reduce_params& params, const optional_params&) const override;
    JitConstants GetJitConstants(const reduce_params& params) const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ELTWISE,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
