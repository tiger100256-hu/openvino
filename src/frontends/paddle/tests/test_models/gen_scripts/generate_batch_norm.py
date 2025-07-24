# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel, saveModel_v3, is_pir_enabled
import sys
import paddle


def batch_norm1(name : str, x, scale, bias, mean, var, data_layout):
    paddle.enable_static()

    node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
    scale_attr = paddle.ParamAttr(name="scale1", initializer=paddle.nn.initializer.Assign(scale))
    bias_attr = paddle.ParamAttr(name="bias1", initializer=paddle.nn.initializer.Assign(bias))

    out = paddle.static.nn.batch_norm(node_x, epsilon=1e-5,
                                    param_attr=scale_attr,
                                    bias_attr=bias_attr,
                                    moving_mean_name="bn_mean1",
                                    moving_variance_name="bn_variance1",
                                    use_global_stats=True,
                                    data_layout=data_layout)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(paddle.static.default_startup_program())
    paddle.static.global_scope().var("bn_mean1").get_tensor().set(mean, paddle.CPUPlace())
    paddle.static.global_scope().var("bn_variance1").get_tensor().set(var, paddle.CPUPlace())

    outs = exe.run(
        feed={'x': x},
        fetch_list=[out])

    saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def batch_norm1_v3(name : str, x, scale, bias, mean, var, data_layout):
    scale_attr = paddle.ParamAttr(name="scale1", initializer=paddle.nn.initializer.Assign(scale))
    bias_attr = paddle.ParamAttr(name="bias1", initializer=paddle.nn.initializer.Assign(bias))
    batch_norm = paddle.nn.BatchNorm(num_channels = scale.shape[0],
                                     epsilon=1e-5,
                                     param_attr=scale_attr,
                                     bias_attr=bias_attr,
                                     moving_mean_name="bn_mean1",
                                     moving_variance_name="bn_variance1",
                                     use_global_stats=True,
                                     data_layout=data_layout)
    batch_norm._parameters["_mean"].set_value(mean)
    batch_norm._parameters["_variance"].set_value(var)

    output = saveModel_v3(name, batch_norm, [x], sys.argv[1])
    return output.numpy()

def batch_norm2(name : str, x, scale, bias, mean, var, data_layout):
    import paddle
    paddle.enable_static()

    node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
    scale_attr = paddle.ParamAttr(name="scale2", initializer=paddle.nn.initializer.Assign(scale))
    bias_attr = paddle.ParamAttr(name="bias2", initializer=paddle.nn.initializer.Assign(bias))

    out = paddle.static.nn.batch_norm(node_x, epsilon=1e-5,
                                    param_attr=scale_attr,
                                    bias_attr=bias_attr,
                                    moving_mean_name="bn_mean2",
                                    moving_variance_name="bn_variance2",
                                    use_global_stats=True,
                                    data_layout=data_layout)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(paddle.static.default_startup_program())
    paddle.static.global_scope().var("bn_mean2").get_tensor().set(mean, paddle.CPUPlace())
    paddle.static.global_scope().var("bn_variance2").get_tensor().set(var, paddle.CPUPlace())

    outs = exe.run(
        feed={'x': x},
        fetch_list=[out])

    saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def batch_norm2_v3(name : str, x, scale, bias, mean, var, data_layout):
    scale_attr = paddle.ParamAttr(name="scale2", initializer=paddle.nn.initializer.Assign(scale))
    bias_attr = paddle.ParamAttr(name="bias2", initializer=paddle.nn.initializer.Assign(bias))
    batch_norm = paddle.nn.BatchNorm(num_channels = scale.shape[0],
                                     epsilon=1e-5,
                                     param_attr=scale_attr,
                                     bias_attr=bias_attr,
                                     moving_mean_name="bn_mean2",
                                     moving_variance_name="bn_variance2",
                                     use_global_stats=True,
                                     data_layout=data_layout)

    batch_norm._parameters["_mean"].set_value(mean)
    batch_norm._parameters["_variance"].set_value(var)

    output = saveModel_v3(name, batch_norm, [x], sys.argv[1])
    return output.numpy()

def main():
    data = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
    # data layout is NCHW
    scale = np.array([1.0, 1.5]).astype(np.float32)
    bias = np.array([0, 1]).astype(np.float32)
    mean = np.array([0, 3]).astype(np.float32)
    var = np.array([1, 1.5]).astype(np.float32)
    if is_pir_enabled():
        batch_norm1_v3("batch_norm_nchw", data, scale, bias, mean, var, "NCHW")
    else:
        batch_norm1("batch_norm_nchw", data, scale, bias, mean, var, "NCHW")

    # data layout is NHWC
    scale = np.array([1.0, 1.5, 2.0]).astype(np.float32)
    bias = np.array([0, 1, 2]).astype(np.float32)
    mean = np.array([0.5, 1.5, 1.5]).astype(np.float32)
    var = np.array([1, 1.5, 2]).astype(np.float32)
    if is_pir_enabled():
        batch_norm2_v3("batch_norm_nhwc", data, scale, bias, mean, var, "NHWC")
    else:
        batch_norm2("batch_norm_nhwc", data, scale, bias, mean, var, "NHWC")

if __name__ == "__main__":
    main()
