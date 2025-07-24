# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# group norm paddle model generator
#
import numpy as np
from save_model import saveModel, saveModel_v3, is_pir_enabled
import paddle
import sys
import os

data_type = "float32"

def group_norm(name: str, x, groups, epsilon, scale, bias, data_layout):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        if scale is False:
            scale_attr = scale
        else:
            scale_attr = paddle.ParamAttr(name="scale1", initializer=paddle.nn.initializer.Assign(scale))
        if bias is False:
            bias_attr = bias
        else:
            bias_attr = paddle.ParamAttr(name="bias1", initializer=paddle.nn.initializer.Assign(bias))

        out = paddle.static.nn.group_norm(node_x, groups=groups,
                                          epsilon=epsilon,
                                          param_attr=scale_attr,
                                          bias_attr=bias_attr,
                                          data_layout=data_layout)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x}, fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def group_norm_v3(name: str, x, groups, epsilon, scale, bias, data_layout):
    if scale is False:
        scale_attr = scale
    else:
        scale_name = name + "_scale1"
        scale_attr = paddle.ParamAttr(name=scale_name, initializer=paddle.nn.initializer.Assign(scale))
    if bias is False:
        bias_attr = bias
    else:
        bias_name = name + "_bias1"
        bias_attr = paddle.ParamAttr(name=bias_name, initializer=paddle.nn.initializer.Assign(bias))

    group_norm_layer = paddle.nn.GroupNorm(num_channels = 4, num_groups=groups,
                                      epsilon=epsilon,
                                      weight_attr=scale_attr,
                                      bias_attr=bias_attr,
                                      data_format=data_layout)
    output = saveModel_v3(name, group_norm_layer, [x], sys.argv[1])
    return output.numpy()

def main():
    enable_pir = False;

    # data layout is NCHW
    data = np.random.random((2, 4, 3, 4)).astype(np.float32)
    groups = 2
    epsilon = 1e-05
    scale = np.random.random(4).astype(np.float32)
    bias = np.random.random(4).astype(np.float32)
    if is_pir_enabled():
        group_norm_v3("group_norm_1", data, groups, epsilon, scale, bias, "NCHW")
    else:
        group_norm("group_norm_1", data, groups, epsilon, scale, bias, "NCHW")

    # data layout is NHWC
    data = np.random.random((2, 4, 3, 4)).astype(np.float32)
    groups = 2
    epsilon = 1e-05
    scale = np.random.random(4).astype(np.float32)
    bias = np.random.random(4).astype(np.float32)
    if is_pir_enabled():
        group_norm_v3("group_norm_2", data, groups, epsilon, scale, bias, "NHWC")
    else:
        group_norm("group_norm_2", data, groups, epsilon, scale, bias, "NHWC")

    # scale and bias are None
    scale = False
    bias = False
    if is_pir_enabled():
        group_norm_v3("group_norm_3", data, groups, epsilon, scale, bias, "NHWC")
    else:
        group_norm("group_norm_3", data, groups, epsilon, scale, bias, "NHWC")


if __name__ == "__main__":
    main()
