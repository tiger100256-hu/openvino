# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np
import paddle

from save_model import exportModel, saveModel_v3, is_pir_enabled
if not is_pir_enabled() :
    class Assign(paddle.nn.Layer):
        def __init__(self):
            super(Assign, self).__init__()
        def forward(self, array):
            result1 = paddle.zeros(shape=[3, 2], dtype='float32')
            paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
            return result1
    model = Assign()
    name = "assign"
    x = np.array([[1, 1],
                [3, 4],
                [1, 3]]).astype(np.int64)
    output = saveModel_v3(name, model, [x], sys.argv[1])

if is_pir_enabled():
    class Assign_none(paddle.nn.Layer):
        def __init__(self):
            super(Assign_none, self).__init__()
        def forward(self, data):
            result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
            return result2
    model = Assign_none()
    name = "assign_none"
    x = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float32') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    output = saveModel_v3(name, model, [x], sys.argv[1])
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
