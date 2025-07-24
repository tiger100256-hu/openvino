# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from save_model import saveModel
import sys
import os

def paddle_matmul(name, x1, x2, x_transpose=False, y_transpose=False):
    import paddle
    enable_pir = False;
    if os.getenv('FLAGS_enable_pir_api') == '1':
        enable_pir = True
    elif os.getenv('FLAGS_enable_pir_api') == '0':
        enable_pir = False
    else:
        enable_pir = False

    if paddle.__version__ >= '3.0.0' and enable_pir :
        class PaddleMul(paddle.nn.Layer):
            def __init__(self, num_channels):
                super(PaddleMul, self).__init__()
                self.batch_norm = paddle.nn.BatchNorm(num_channels, use_global_stats=True)
            def forward(self, x1, x2):
                mul_node = paddle.matmul(x1, x2, x_transpose, y_transpose)
                result = self.batch_norm(mul_node)
                return result
        if y_transpose:
            num_channels = x2.shape[0]
        else:
            num_channels = x2.shape[1]
        model = PaddleMul(num_channels)
        net = paddle.jit.to_static(model, full_graph=True)
        net.eval()
        model_dir = os.path.join(sys.argv[1], name)
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        np.save(os.path.join(model_dir, "input0"), x1)
        np.save(os.path.join(model_dir, "input1"), x2)
        input_tensor0 = paddle.to_tensor(x1)
        input_tensor1 = paddle.to_tensor(x2)
        output0 = net(input_tensor0, input_tensor1)
        np.save(os.path.join(model_dir, "output0"), output0.numpy())
        input_spec = [paddle.static.InputSpec(shape=x1.shape, dtype=x1.dtype),
                      paddle.static.InputSpec(shape=x2.shape, dtype=x2.dtype)]
        paddle.jit.save(net, model_path, input_spec)
        return output0.numpy()

    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x1 = paddle.static.data(name='x1', shape=x1.shape, dtype=x1.dtype)
        node_x2 = paddle.static.data(name='x2', shape=x2.shape, dtype=x2.dtype)
        if paddle.__version__ >= '2.0.0':
            mul_node = paddle.matmul(node_x1, node_x2, x_transpose, y_transpose)
        else:
            mul_node = paddle.fluid.layers.matmul(node_x1, node_x2, x_transpose, y_transpose)
        result = paddle.static.nn.batch_norm(mul_node, use_global_stats=True)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x1': x1, 'x2': x2},
            fetch_list=[result])
        saveModel(name, exe, feed_vars=[node_x1, node_x2], fetchlist=[result], inputs=[x1, x2], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


if __name__ == "__main__":
    input_2x5 = np.array([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10]]).astype(np.float32)

    input_5x3 = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12],
                       [13, 14, 15]]).astype(np.float32)

    input_5x2 = np.array([[1, 2],
                          [4, 5],
                          [7, 8],
                          [10, 11],
                          [13, 14]]).astype(np.float32)

    input_2x3 = np.array([[1, 2, 3],
                          [4, 5, 6]]).astype(np.float32)

    paddle_matmul("matmul_xt", input_2x5, input_2x3, x_transpose=True, y_transpose=False)
    paddle_matmul("matmul_yt", input_2x3, input_5x3, x_transpose=False, y_transpose=True)
    paddle_matmul("matmul_xt_yt", input_2x5, input_5x2, x_transpose=True, y_transpose=True)
