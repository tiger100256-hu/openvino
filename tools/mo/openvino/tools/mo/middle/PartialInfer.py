# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, shape_array, \
    dynamic_dimension_value
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.parameter import Parameter


class PartialInfer(MiddleReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from openvino.tools.mo.front.create_tensor_nodes import CreateTensorNodes
        return [CreateTensorNodes]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        dynamic_inputs = {}
        for parameter in graph.get_op_nodes(op='Parameter'):
            param_shape = parameter.soft_get('shape', shape_array(dynamic_dimension_value))
            if not is_fully_defined(param_shape):
                parameter_name = parameter.soft_get('name', parameter.id)
                dynamic_inputs[parameter_name] = parameter
        if dynamic_inputs:
            log.error('The model contains input(s) with partially defined shapes: {}. '
                      'Starting from the 2022.1 release the Model Optimizer can generate an IR with partially defined '
                      'input shapes ("-1" dimension in the TensorFlow model or dimension with string value in the ONNX '
                      'model). Some of the OpenVINO plugins require model input shapes to be static, so you should '
                      'call "reshape" method in the Inference Engine and specify static input shapes. For optimal '
                      'performance, it is still recommended to update input shapes with fixed ones using "--input" or '
                      '"--input_shape" command-line parameters.'
                      .format(','.join('name="{}" shape="{}"'.format(name, Parameter.shape_serialize(parameter))
                                       for name, parameter in dynamic_inputs.items())),
                      extra={'is_warning': True})
        partial_infer(graph)
