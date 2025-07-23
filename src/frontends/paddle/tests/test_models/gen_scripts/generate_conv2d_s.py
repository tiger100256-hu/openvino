# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import paddle
import numpy as np
import os
import sys

from save_model import saveModel

enable_pir = False;
if os.getenv('FLAGS_enable_pir_api') == '1':
    enable_pir = True
elif os.getenv('FLAGS_enable_pir_api') == '0':
    enable_pir = False
else:
    enable_pir = False

if paddle.__version__ >= '3.0.0' and enable_pir:
    conv_layer = paddle.nn.Conv2D(
        in_channels=3,      # Number of input channels (e.g., RGB image has 3)
        out_channels=5,    # Number of output channels (filters)
        kernel_size=1,      # Size of the convolution kernel
        stride=1,           # Stride of the convolution
        padding=1,           # Padding added to both sides of the input
        dilation=1,
        groups=1,
        bias_attr=None
    )
    net = paddle.jit.to_static(conv_layer, full_graph=True)
    net.eval()
    x = np.random.rand(1, 3, 4, 4).astype('float32');
    name = "conv2d_s"
    model_dir = os.path.join(sys.argv[1], name)
    model_path = os.path.join(model_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.save(os.path.join(model_dir, "input0"), x)
    input_tensor = paddle.to_tensor(x)
    output = net(input_tensor)
    np.save(os.path.join(model_dir, "output0"), output.numpy())
    input_spec = [paddle.static.InputSpec(shape=[1,3,4,4], dtype='float32')]
    paddle.jit.save(net, model_path, input_spec)
    sys.exit(0)


if paddle.__version__ >= '2.6.0':
    import paddle.base as fluid
else:
    from paddle import fluid

paddle.enable_static()

inp_blob = np.random.randn(1, 3, 4, 4).astype(np.float32)

if paddle.__version__ >= '2.0.0':
    x = paddle.static.data(name='x', shape=[1, 3, 4, 4], dtype='float32')
    test_layer = paddle.static.nn.conv2d(input=x, num_filters=5, filter_size=(1, 1), stride=(1, 1), padding=(1, 1),
                                         dilation=(1, 1), groups=1, bias_attr=False)
else:
    x = fluid.data(name='x', shape=[1, 3, 4, 4], dtype='float32')
    test_layer = fluid.layers.conv2d(input=x, num_filters=5, filter_size=(1, 1), stride=(1, 1), padding=(1, 1),
                                     dilation=(1, 1), groups=1, bias_attr=False)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
inp_dict = {'x': inp_blob}
var = [test_layer]
res_paddle = exe.run(fluid.default_main_program(),
                     fetch_list=var, feed=inp_dict)

saveModel(os.path.join(sys.argv[1], "conv2d_s", "conv2d_s"), exe, feed_vars=[x], fetchlist=var, inputs=[inp_blob], outputs=[res_paddle[0]], target_dir=sys.argv[1])
