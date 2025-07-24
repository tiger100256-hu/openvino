# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import paddle

from save_model import exportModel
enable_pir = False;
if os.getenv('FLAGS_enable_pir_api') == '1':
    enable_pir = True
elif os.getenv('FLAGS_enable_pir_api') == '0':
    enable_pir = False
else:
    enable_pir = False

#if paddle.__version__ >= '3.0.0' and enable_pir:
if paddle.__version__ >= '3.0.0' and not enable_pir :
    class Assign(paddle.nn.Layer):
        def __init__(self):
            super(Assign, self).__init__()
        def forward(self, array):
            result1 = paddle.zeros(shape=[3, 2], dtype='float32')
            paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
            return result1
    model = Assign()
    net = paddle.jit.to_static(model, full_graph=True)
    net.eval()
    name = "assign"
    model_dir = os.path.join(sys.argv[1], name)
    model_path = os.path.join(model_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    array = np.array([[1, 1],
                  [3, 4],
                  [1, 3]]).astype(np.int64)
    np.save(os.path.join(model_dir, "input0"), array)
    input_tensor0 = paddle.to_tensor(array)
    output0 = net(input_tensor0)
    np.save(os.path.join(model_dir, "output0"), output0.numpy())
    input_spec = [paddle.static.InputSpec(shape=[3,2], dtype='int64')]
    paddle.jit.save(net, model_path, input_spec)

if paddle.__version__ >= '3.0.0' and enable_pir:
    class Assign_none(paddle.nn.Layer):
        def __init__(self):
            super(Assign_none, self).__init__()
        def forward(self, data):
            result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
            return result2
    model = Assign_none()
    net = paddle.jit.to_static(model, full_graph=True)
    net.eval()
    name = "assign_none"
    model_dir = os.path.join(sys.argv[1], name)
    model_path = os.path.join(model_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float32') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    np.save(os.path.join(model_dir, "input0"), data)
    input_tensor0 = paddle.to_tensor(data)
    output0 = net(input_tensor0)
    np.save(os.path.join(model_dir, "output0"), output0.numpy())
    input_spec = [paddle.static.InputSpec(shape=[3,2], dtype='float32')]
    paddle.jit.save(net, model_path, input_spec)
    sys.exit(0)

'''
assign w/ output
'''
@paddle.jit.to_static
def test_assign_output(array):
    result1 = paddle.zeros(shape=[3, 2], dtype='float32')
    paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
    return result1

array = np.array([[1, 1],
                [3, 4],
                [1, 3]]).astype(np.int64)
exportModel('assign_output', test_assign_output, [array], target_dir=sys.argv[1])

'''
assign w/o output
'''
@paddle.jit.to_static
def test_assign_none(data):
    result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    return result2

data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float32') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
exportModel('assign_none', test_assign_none, [data], target_dir=sys.argv[1])
