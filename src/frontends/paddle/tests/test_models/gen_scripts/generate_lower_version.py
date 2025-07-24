# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import paddle
import numpy as np
import os
import sys
enable_pir = False;
if os.getenv('FLAGS_enable_pir_api') == '1':
    enable_pir = True
elif os.getenv('FLAGS_enable_pir_api') == '0':
    enable_pir = False
else:
    enable_pir = False

if paddle.__version__ >= '3.0.0' and enable_pir:
    sys.exit(0)
elif paddle.__version__ >= '2.6.0':
    from paddle.base.proto import framework_pb2
else:
    from paddle.fluid.proto import framework_pb2

paddle.enable_static()

inp_blob = np.random.randn(1, 3, 4, 4).astype(np.float32)
print(sys.path)
main_program = paddle.static.Program()
startup_program = paddle.static.Program()

with paddle.static.program_guard(main_program, startup_program):
    x = paddle.static.data(name='x', shape=[1, 3, 4, 4], dtype='float32')
    test_layer = paddle.static.nn.conv2d(input=x, num_filters=5, filter_size=(1, 1), stride=(1, 1), padding=(1, 1),
                                     dilation=(1, 1), groups=1, bias_attr=False)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    exe.run(startup_program)
    inp_dict = {'x': inp_blob}
    var = [test_layer]
    res_paddle = exe.run(paddle.static.default_main_program(), fetch_list=var, feed=inp_dict)
    paddle.static.save_inference_model(os.path.join(sys.argv[1], "lower_version/", "lower_version"), [x], [test_layer], exe, program=main_program)


fw_model = framework_pb2.ProgramDesc()
with open(os.path.join(sys.argv[1], "lower_version", "lower_version.pdmodel"), mode='rb') as file:
    fw_model.ParseFromString(file.read())

fw_model.version.version = 1800000
print(fw_model.version.version)
with open(os.path.join(sys.argv[1], "lower_version", "lower_version.pdmodel"), "wb") as f:
    f.write(fw_model.SerializeToString())




