# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# partial_concat paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys
import os

def partial_concat(name: str, x, y, start_index=0, length=-1):
    enable_pir = False;
    if os.getenv('FLAGS_enable_pir_api') == '1':
        enable_pir = True
    elif os.getenv('FLAGS_enable_pir_api') == '0':
        enable_pir = False
    else:
        enable_pir = False

    if paddle.__version__ >= '3.0.0' and enable_pir:
        class PartialConcat(paddle.nn.Layer):
            def __init__(self, start_index, length):
                super().__init__()
                self.start_index = start_index
                self.length = length
            def forward(self, x, y):
                sliced = []
                axis = 0
                for item in [x , y]:
                    end_index = x.shape[axis] if length == -1 else start_index + length
                    sliced.append(item.slice([axis], [start_index], [end_index]))
                return paddle.concat(sliced, axis=axis)
        model = PartialConcat(start_index=start_index, length=length)
        net = paddle.jit.to_static(model, full_graph=True)
        net.eval()
        model_dir = os.path.join(sys.argv[1], name)
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        np.save(os.path.join(model_dir, "input0"), x)
        np.save(os.path.join(model_dir, "input1"), y)
        input_tensor0 = paddle.to_tensor(x)
        input_tensor1 = paddle.to_tensor(y)
        output = net(input_tensor0, input_tensor1)
        np.save(os.path.join(model_dir, "output0"), output.numpy())
        input_spec = [paddle.static.InputSpec(shape=x.shape, dtype=x.dtype),
                      paddle.static.InputSpec(shape=y.shape, dtype=y.dtype)]
        paddle.jit.save(net, model_path, input_spec)
        return output.numpy()

    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_data = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        y_data = paddle.static.data(name="y", shape=x.shape, dtype=y.dtype)

        if paddle.__version__ >= '2.5.1':
            out = paddle.incubate.layers.nn.partial_concat(
                [x_data, y_data], start_index=start_index, length=length
            )
        else:
            out = paddle.fluid.contrib.layers.partial_concat(
            [x_data, y_data], start_index=start_index, length=length
            )

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x, "y": y}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=[x_data, y_data],
            fetchlist=[out],
            inputs=[x, y],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    dtype = 'float32'
    x = np.random.randn(6, 4).astype(dtype)
    y = np.random.randn(6, 4).astype(dtype)
    partial_concat("partial_concat_1", x, y, start_index=2, length=2)


    dtype = 'int32'
    x = np.random.randint(-10, 10, [5, 3]).astype(dtype)
    y = np.random.randint(-10, 10, [5, 3]).astype(dtype)
    partial_concat("partial_concat_2", x, y, start_index=1, length=-1)

    dtype = 'int64'
    x = np.random.randint(-10, 10, [8, 10]).astype(dtype)
    y = np.random.randint(-10, 10, [8, 10]).astype(dtype)
    partial_concat("partial_concat_3", x, y, start_index=1, length=5)

if __name__ == "__main__":
    main()
