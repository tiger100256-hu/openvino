# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from save_model import saveModel
import numpy as np
import paddle
import sys
import os

def run_and_save_model(input_x, name, feed, fetch_list, main_prog, start_prog):
    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    exe.run(start_prog)
    outs = exe.run(
        feed={'x': input_x},
        fetch_list=fetch_list,
        program=main_prog)

    with paddle.static.program_guard(main_prog, start_prog):
        saveModel(name, exe, feed_vars=[feed], fetchlist=fetch_list, inputs=[input_x],
                  outputs=[outs[0]], target_dir=sys.argv[1])


def paddle_conv2d(input_x, name, input_shape, kernel, dilation, padding, stride, groups=1, use_cudnn=True):
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        data = paddle.static.data(name='x', shape=input_shape, dtype='float32')
        weight_attr = paddle.ParamAttr(name="conv2d_weight", initializer=paddle.nn.initializer.Assign(kernel))
        conv2d = paddle.static.nn.conv2d(input=data, num_filters=kernel.shape[0], filter_size=kernel.shape[2:4],
                                       padding=padding, param_attr=weight_attr, dilation=dilation, stride=stride, groups=groups, use_cudnn=use_cudnn)
    run_and_save_model(input_x, name, data, conv2d, main_program, startup_program)

def paddle_conv2d_v3(input_x, name, input_shape, kernel, dilation, padding, stride, groups=1, use_cudnn=True):
    conv2d_weight_param_name = "conv2d_weight" + name
    weight_attr = paddle.ParamAttr(name=conv2d_weight_param_name, initializer=paddle.nn.initializer.Assign(kernel))
    conv_layer = paddle.nn.Conv2D(
        in_channels=3,      # Number of input channels (e.g., RGB image has 3)
        out_channels=kernel.shape[0],    # Number of output channels (filters)
        kernel_size=kernel.shape[2:4],      # Size of the convolution kernel
        stride=stride,           # Stride of the convolution
        padding=padding,           # Padding added to both sides of the input
        dilation=dilation,
        groups=groups,
        weight_attr=weight_attr,
        bias_attr=None
    )
    net = paddle.jit.to_static(conv_layer, full_graph=True)
    net.eval()
    x = np.random.rand(*input_shape).astype('float32');
    model_dir = os.path.join(sys.argv[1], name)
    model_path = os.path.join(model_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.save(os.path.join(model_dir, "input0"), x)
    input_tensor = paddle.to_tensor(x)
    output = net(input_tensor)
    np.save(os.path.join(model_dir, "output0"), output.numpy())
    input_spec = [paddle.static.InputSpec(shape=input_shape, dtype='float32')]
    paddle.jit.save(net, model_path, input_spec)

if __name__ == "__main__":

    test_cases =[
        {
            "input_x": np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                                   [5., 6., 7., 8., 9.],
                                   [10., 11., 12., 13., 14.],
                                   [15., 16., 17., 18., 19.],
                                   [20., 21., 22., 23., 24.],
                                   [25., 26., 27., 28., 29.],
                                   [30., 31., 32., 33., 34.,]]]]).astype(np.float32),
            "name": "conv2d_SAME_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array([[[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]]]]).astype(np.float32),
            "dilation": 1,
            "padding": "SAME",
            "stride" : 2,
        },
        {
            "input_x": np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                                   [5., 6., 7., 8., 9.],
                                   [10., 11., 12., 13., 14.],
                                   [15., 16., 17., 18., 19.],
                                   [20., 21., 22., 23., 24.],
                                   [25., 26., 27., 28., 29.],
                                   [30., 31., 32., 33., 34.,]]]]).astype(np.float32),
            "name": "conv2d_VALID_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array([[[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]]]]).astype(np.float32),
            "dilation": 1,
            "padding": "VALID",
            "stride" : 2,
        },
        {
            "input_x": np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                                   [5., 6., 7., 8., 9.],
                                   [10., 11., 12., 13., 14.],
                                   [15., 16., 17., 18., 19.],
                                   [20., 21., 22., 23., 24.],
                                   [25., 26., 27., 28., 29.],
                                   [30., 31., 32., 33., 34.,]]]]).astype(np.float32),
            "name": "conv2d_strides_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array([[[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]]]]).astype(np.float32),
            "dilation": 1,
            "padding": 1,
            "stride" : 2,
        },
        {   "input_x": np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                                   [5., 6., 7., 8., 9.],
                                   [10., 11., 12., 13., 14.],
                                   [15., 16., 17., 18., 19.],
                                   [20., 21., 22., 23., 24.],
                                   [25., 26., 27., 28., 29.],
                                   [30., 31., 32., 33., 34.,]]]]).astype(np.float32),
            "name": "conv2d_strides_no_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array([[[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]]]]).astype(np.float32),
            "dilation": 1,
            "padding": 0,
            "stride" : 2,
            },
        {   "input_x": np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                                   [5., 6., 7., 8., 9.],
                                   [10., 11., 12., 13., 14.],
                                   [15., 16., 17., 18., 19.],
                                   [20., 21., 22., 23., 24.],
                                   [25., 26., 27., 28., 29.],
                                   [30., 31., 32., 33., 34.,]]]]).astype(np.float32),
            "name": "conv2d_strides_assymetric_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array([[[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]]]]).astype(np.float32),
            "dilation": 1,
            "padding": [1,1,0,1],
            "stride" : 2,
            },
        {
            "input_x": np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                                   [5., 6., 7., 8., 9.],
                                   [10., 11., 12., 13., 14.],
                                   [15., 16., 17., 18., 19.],
                                   [20., 21., 22., 23., 24.],
                                   [25., 26., 27., 28., 29.],
                                   [30., 31., 32., 33., 34.,]]]]).astype(np.float32),
            "name": "conv2d_dilation_assymetric_pads_strides",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array([[[[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]]]]).astype(np.float32),
            "dilation": 1,
            "padding": [1, 1, 1, 2],
            "stride" : [3, 1],
        },
        {
            "input_x": np.arange(27).astype(np.float32).reshape([1, 3, 3, 3]),
            "name": "depthwise_conv2d_convolution",
            "input_shape": [1, 3, 3, 3],
            "kernel": np.ones([3, 1, 3, 3]).astype(np.float32),
            "dilation": 1,
            "padding": 1,
            "stride": 1,
            "groups": 3,
            "use_cudnn": False
        }
    ]
    enable_pir = False;
    if os.getenv('FLAGS_enable_pir_api') == '1':
        enable_pir = True
    elif os.getenv('FLAGS_enable_pir_api') == '0':
        enable_pir = False
    else:
        enable_pir = False

    for test in test_cases:
        if paddle.__version__ >= '3.0.0' and enable_pir:
            paddle_conv2d_v3(test['input_x'], test['name'], test["input_shape"],
                             test['kernel'], test['dilation'],
                             test['padding'],
                             test['stride'],
                             1 if "groups" not in test else test['groups'],
                             True if "use_cudnn" not in test else test['use_cudnn'])
        else:
            paddle_conv2d(test['input_x'], test['name'], test["input_shape"],
                        test['kernel'], test['dilation'],
                        test['padding'],
                        test['stride'],
                        1 if "groups" not in test else test['groups'],
                        True if "use_cudnn" not in test else test['use_cudnn'])


