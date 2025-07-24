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
    class TwoInputAndTwoOutput(paddle.nn.Layer):
        def __init__(self):
            super(TwoInputAndTwoOutput, self).__init__()
            self.conv_layer1 = paddle.nn.Conv2D(
                in_channels=1,      # Number of input channels (e.g., RGB image has 3)
                out_channels=1,    # Number of output channels (filters)
                kernel_size=1,      # Size of the convolution kernel
                stride=1,           # Stride of the convolution
                padding=0,           # Padding added to both sides of the input
                dilation=1,
                groups=1,
                bias_attr=None
            )
            self.conv_layer2 = paddle.nn.Conv2D(
                in_channels=2,      # Number of input channels (e.g., RGB image has 3)
                out_channels=1,    # Number of output channels (filters)
                kernel_size=1,      # Size of the convolution kernel
                stride=1,           # Stride of the convolution
                padding=0,           # Padding added to both sides of the input
                dilation=1,
                groups=1,
                bias_attr=None
            )
            self.relu2a = paddle.nn.ReLU()
            self.relu2b = paddle.nn.ReLU()
            self.relu3a = paddle.nn.ReLU()
            self.relu3b = paddle.nn.ReLU()
        def forward(self, x, y):
            conv1_res = self.conv_layer1(x)
            conv2_res = self.conv_layer2(y)
            add1_res = paddle.add(conv1_res, conv2_res)
            relu2a_res = self.relu2a(add1_res)
            relu2b_res = self.relu2b(add1_res)
            add2_res = paddle.add(relu2a_res, relu2b_res)
            relu3a_res = self.relu3a(add2_res)
            relu3b_res = self.relu3b(add2_res)
            return relu3a_res, relu3b_res
    model = TwoInputAndTwoOutput()
    net = paddle.jit.to_static(model, full_graph=True)
    net.eval()
    x = np.random.rand(1, 1, 3, 3).astype('float32');
    y = np.random.rand(1, 2, 3, 3).astype('float32');
    name = "2in_2out"
    model_dir = os.path.join(sys.argv[1], name)
    model_path = os.path.join(model_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.save(os.path.join(model_dir, "input0"), x)
    np.save(os.path.join(model_dir, "input1"), y)
    input_tensor0 = paddle.to_tensor(x)
    input_tensor1 = paddle.to_tensor(y)
    output0, output1 = net(input_tensor0, input_tensor1)
    np.save(os.path.join(model_dir, "output0"), output0.numpy())
    np.save(os.path.join(model_dir, "output1"), output1.numpy())
    input_spec = [paddle.static.InputSpec(shape=[1,1,3,3], dtype='float32'),
                  paddle.static.InputSpec(shape=[1,2,3,3], dtype='float32')]
    paddle.jit.save(net, model_path, input_spec)
    sys.exit(0)

if paddle.__version__ >= '2.6.0':
    import paddle.base as fluid
else:
    from paddle import fluid

paddle.enable_static()

inp_blob1 = np.random.randn(1, 1, 3, 3).astype(np.float32)
inp_blob2 = np.random.randn(1, 2, 3, 3).astype(np.float32)

if paddle.__version__ >= '2.0.0':
    x1 = paddle.static.data(name='inputX1', shape=[1, 1, 3, 3], dtype='float32')
    x2 = paddle.static.data(name='inputX2', shape=[1, 2, 3, 3], dtype='float32')
else:
    x1 = fluid.data(name='inputX1', shape=[1, 1, 3, 3], dtype='float32')
    x2 = fluid.data(name='inputX2', shape=[1, 2, 3, 3], dtype='float32')

if paddle.__version__ >= '2.0.0':
    conv2d1 = paddle.static.nn.conv2d(input=x1, num_filters=1, filter_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                    dilation=(1, 1), groups=1, bias_attr=False, name="conv2dX1")

    conv2d2 = paddle.static.nn.conv2d(input=x2, num_filters=1, filter_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                dilation=(1, 1), groups=1, bias_attr=False, name="conv2dX2")

    add1 = paddle.add(conv2d1, conv2d2, name="add1.tmp_0")

    relu2a = paddle.nn.functional.relu(add1, name="relu2a")
    relu2b = paddle.nn.functional.relu(add1, name="relu2b")

    add2 = paddle.add(relu2a, relu2b, name="add2.tmp_0")

    relu3a = paddle.nn.functional.relu(add2, name="relu3a")
    relu3b = paddle.nn.functional.relu(add2, name="relu3b")
else:
    conv2d1 = fluid.layers.conv2d(input=x1, num_filters=1, filter_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                 dilation=(1, 1), groups=1, bias_attr=False, name="conv2dX1")

    conv2d2 = fluid.layers.conv2d(input=x2, num_filters=1, filter_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                dilation=(1, 1), groups=1, bias_attr=False, name="conv2dX2")

    add1 = fluid.layers.elementwise_add(conv2d1, conv2d2, name="add1")

    relu2a = fluid.layers.relu(add1, name="relu2a")
    relu2b = fluid.layers.relu(add1, name="relu2b")

    add2 = fluid.layers.elementwise_add(relu2a, relu2b, name="add2")

    relu3a = fluid.layers.relu(add2, name="relu3a")
    relu3b = fluid.layers.relu(add2, name="relu3b")

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
inp_dict = {'inputX1': inp_blob1, 'inputX2': inp_blob2}
var = [relu3a, relu3b]
res_paddle = exe.run(fluid.default_main_program(), fetch_list=var, feed=inp_dict)

saveModel("2in_2out", exe, feed_vars=[x1, x2],
          fetchlist=var,
          inputs=[inp_blob1, inp_blob2],
          outputs=[res_paddle[0], res_paddle[1]], target_dir=sys.argv[1])
