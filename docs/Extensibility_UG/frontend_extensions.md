﻿# Frontend Extensions {#openvino_docs_Extensibility_UG_Frontend_Extensions}

@sphinxdirective

The goal of this chapter is to explain how to use Frontend extension classes to facilitate 
mapping of custom operations from framework model representation to OpenVINO representation. 
Refer to :doc:`Introduction to OpenVINO Extension <openvino_docs_Extensibility_UG_Intro>` to 
understand the entire flow.

This API is applicable to new frontends only, which exist for ONNX, TensorFlow Lite, PaddlePaddle, and TensorFlow. 
If a different model format is used, follow legacy 
:doc:`Model Optimizer Extensions <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer>` 
guide.

.. note:: 

   This documentation is written based on the `Template extension <https://github.com/openvinotoolkit/openvino/tree/master/src/core/template_extension/new>`__, 
   which demonstrates extension development details based on minimalistic ``Identity`` 
   operation that is a placeholder for your real custom operation. You can review the complete code, 
   which is fully compilable, to see how it works.

Single Operation Mapping with OpExtension
#########################################

This section covers the case when a single operation in framework representation is mapped to a single 
operation in OpenVINO representation. This is called *one-to-one mapping*. There is ``OpExtension`` 
class that works well if all the following conditions are satisfied:

1. Number of inputs to operation in the Framework representation is the same as in the OpenVINO representation.
2. Number of outputs is also the same in both representations.
3. Inputs can be indexed and are mapped in order correspondingly, e.g. 
   input with index 0 in framework representation maps to input with index 0 in OpenVINO representation and so on.
4. The same for outputs.
5. Each attribute in OpenVINO operation can be initialized from one of the attributes of original operation or by 
   some predefined constant value. Value of copied attributes cannot contain expressions, value is accepted as-is, 
   so type of a value should be compatible.

.. note::

   ``OpExtension`` class is currently available for ONNX and TensorFlow frontends. 
   PaddlePaddle frontend has named inputs and outputs for operation (not indexed) 
   therefore OpExtension mapping is not applicable for this case.

The following example maps ONNX operation with the type of `Identity <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity>`__ 
to OpenVINO template extension ``Identity`` class.

.. doxygensnippet:: docs/snippets/ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_Identity_header]

.. doxygensnippet:: docs/snippets/ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_Identity]

The mapping doesn’t involve any attributes, as operation Identity doesn’t have them.

Extension objects, like just constructed ``extension`` can be used to add to the 
OpenVINO runtime just before the loading a model that contains custom operations:

.. doxygensnippet:: docs/snippets/ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_read_model]

Or extensions can be constructed in a separately compiled shared library. 
Separately compiled library can be used in Model Optimizer or ``benchmark_app``. 
Read about how to build and load such a library in the chapter of “Create library with extensions” in 
:doc:`Introduction to OpenVINO Extension <openvino_docs_Extensibility_UG_Intro>`.

If operation have multiple inputs and/or outputs they will be mapped in order. 
The type of elements in input/output tensors should match expected types in the surrounding operations. 
For example, if a custom operation produces the ``f32`` data type, the operation that consumes this output 
should also support ``f32``. Otherwise, model conversion fails with an error, as no automatic type conversion is performed.

Converting to Standard OpenVINO Operation
+++++++++++++++++++++++++++++++++++++++++

``OpExtension`` class can be used when mapping to one of the operations from standard OpenVINO 
operation set is what you need and there is no class like ``TemplateExtension::Identity`` implemented.

Here is an example of a custom framework operation 'MyRelu'. Assume it is mathematically equivalent 
to standard ``Relu`` that exists in the OpenVINO operation set, but for some reason has the type name of 'MyRelu'. 
In this case, you can directly say that 'MyRelu' -> ``Relu`` mapping should be used:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/ov_extensions.cpp
         :language: cpp
         :fragment: [frontend_extension_MyRelu]

   .. tab-item:: Python
      :sync: python
   
      .. doxygensnippet:: docs/snippets/ov_extensions.py
         :language: python
         :fragment: [py_frontend_extension_MyRelu]


In the resulting converted OpenVINO model, “MyRelu” operation will be replaced by the standard operation 
``Relu`` from the latest available OpenVINO operation set. Notice that when standard operation is used, 
it can be specified using just a type string (“Relu”) instead of using a ``ov::opset8::Relu`` class name as a 
template parameter for ``OpExtension``. This method is available for operations from the standard operation set only. 
For a user custom OpenVINO operation the corresponding class should be always specified as a template parameter 
as it was demonstrated with ``TemplateExtension::Identity``.

Attribute Mapping
++++++++++++++++++

As described above, ``OpExtension`` is useful when attributes can be mapped one by one or initialized by a constant. 
If the set of attributes in framework representation and OpenVINO representation completely match by their names and types, 
nothing should be specified in OpExtension constructor parameters. The attributes are discovered and mapped 
automatically based on ``visit_attributes`` method that should be defined for any OpenVINO operation.

Imagine you have CustomOperation class implementation that has two attributes with names: ``attr1`` and ``attr2``.

.. doxygensnippet:: docs/snippets/ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_CustomOperation]

And the original model in the framework representation also has operation named “CustomOperation” with the same 
``attr1`` and ``attr2`` attributes. Then with the following code:

.. doxygensnippet:: docs/snippets/ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_CustomOperation_as_is]

Both ``attr1`` and ``attr2`` are copied from framework representation to OpenVINO representation automatically. 
If for some reason names of attributes are different but values still can be copied “as-is” you can pass attribute 
names mapping in ``OpExtension`` constructor:

.. doxygensnippet:: docs/snippets/ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_CustomOperation_rename]

Where ``fw_attr1`` and ``fw_attr2`` are names for corresponding attributes in framework operation representation.

If copying of an attribute is not what you need, ``OpExtension`` also can set attribute to predefined constant value. 
For the same ``CustomOperation``, imagine you want to set ``attr2`` to value 5 instead of copying from ``fw_attr2``, 
to achieve that do the following:

.. doxygensnippet:: docs/snippets/ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_CustomOperation_rename_set]

So the conclusion is that each attribute of target OpenVINO operation should be initialized either by

1. Setting automatically due to name matching
2. Mapped by attribute name
3. Set to a constant value

This is achieved by specifying maps as arguments for `OpExtension` constructor.

Mapping custom operations to frontends with OPENVINO_FRAMEWORK_MAP macro
---------------------------------------------------------------------------

.. note:: 
   
   Below solution works only for ONNX and Tensorflow frontends.

``OPENVINO_FRAMEWORK_MAP`` is a macro that should be used inside OpenVINO operation's class definition 
and that lets you specify the mapping between this operation to a frontend operation.

Let's consider the following example. Imagine you have an ONNX model with ``CustomOp`` operation 
(and this operation has ``mode`` attribute) and a Tensorflow model with ``CustomOpV3`` operation 
(this operation has ``axis`` attribute) and both of them can be implemented with a single OpenVINO 
operation ``CustomOp`` like follows:

.. doxygensnippet:: ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_framework_map_macro_headers]

.. doxygensnippet:: ov_extensions.cpp
   :language: cpp
   :fragment: [frontend_extension_framework_map_macro_CustomOp]


Let's take a closer look at the parameters this macro takes:

.. code-block::cpp

   OPENVINO_FRAMEWORK_MAP(framework, name, attributes_map, attributes_values)

- ``framework`` - framework name.
- ``name`` - the framework operation name. It's optional if the OpenVINO custom operation name 
  (that is the name that is passed as the first parameter to `OPENVINO_OP` macro) is the 
  same as the framework operation name and both ``attributes_map`` and ``attributes_values`` are not provided.
- ``attributes_map`` - used to provide a mapping between OpenVINO operation attribute and 
  framework operation attribute. Contains key-value pairs, where key is an OpenVINO operation 
  attribute name and value is its corresponding framework operation attribute name. 
  This parameter is optional if the number of OpenVINO operation attributes and their names 
  match one-to-one with framework operation attributes.
- ``attributes_values`` - used to provide default values for OpenVINO operation attributes 
  that are not specified in ``attributes_map``. Contains key-value pairs, where key is an OpenVINO 
  operation attribute name and the value is this attribute value. This parameter cannot be provided 
  if ``attributes_map`` contains all of OpenVINO operation attributes or if ``attributes_map`` is not provided.

In the example above, ``OPENVINO_FRAMEWORK_MAP`` is used twice.
First, OpenVINO ``CustomOp`` is mapped to ONNX ``CustomOp`` operation, ``m_mode`` attribute is mapped to ``mode`` 
attribute, while ``m_axis`` attribute gets the default value ``-1``. Secondly, OpenVINO `CustomOp` is mapped 
to Tensorflow ``CustomOpV3`` operation, ``m_axis`` attribute is mapped to ``axis`` attribute, while ``m_mode`` 
attribute gets the default value ``linear``.

The last step is to register this custom operation by following:
@snippet ov_extensions.cpp frontend_extension_framework_map_macro_add_extension

Mapping to Multiple Operations with ConversionExtension
#######################################################

Previous sections cover the case when a single operation is mapped to a single operation with optional 
adjustment in names and attribute values. That is likely enough for your own custom operation with existing 
C++ kernel implementation. In this case your framework representation and OpenVINO representation for the 
operation are under your control and inputs/outpus/attributes can be aligned to make ``OpExtension`` usable.

In case if one-to-one mapping is not possible, *decomposition to multiple operations* should be considered. 
It is achieved by using more verbose and less automated ``ConversionExtension`` class. 
It enables writing arbitrary code to replace a single framework operation by multiple connected OpenVINO 
operations constructing dependency graph of any complexity.

``ConversionExtension`` maps a single operation to a function which builds a graph using OpenVINO 
operation classes. Follow chapter :ref:`Build a Model in OpenVINO Runtime <ov_ug_build_model>` to 
learn how to use OpenVINO operation classes to build a fragment of model for replacement.

The next example illustrates using ``ConversionExtension`` for conversion of “ThresholdedRelu” 
from ONNX according to the formula: ``ThresholdedRelu(x, alpha) -> Multiply(x, Convert(Greater(x, alpha), type=float))``.

.. note:: 

   ``ThresholdedRelu`` is one of the standard ONNX operators which is supported by ONNX frontend 
   natively out-of-the-box. Here we are re-implementing it to illustrate how you can add a similar 
   support for your custom operation instead of ``ThresholdedRelu``.

.. tab-set::

   .. tab-item:: C++
      :sync: cpp
 
      .. doxygensnippet:: docs/snippets/ov_extensions.cpp
         :language: cpp
         :fragment: [frontend_extension_ThresholdedReLU_header]

   .. tab-item:: Python
      :sync: python

      .. doxygensnippet:: docs/snippets/ov_extensions.py
         :language: python
         :fragment: [py_frontend_extension_ThresholdedReLU_header]

.. tab-set::

   .. tab-item:: C++
      :sync: cpp
 
      .. doxygensnippet:: docs/snippets/ov_extensions.cpp
         :language: cpp
         :fragment: [frontend_extension_ThresholdedReLU]

   .. tab-item:: Python
      :sync: python
 
      .. doxygensnippet:: docs/snippets/ov_extensions.py
         :language: python
         :fragment: [py_frontend_extension_ThresholdedReLU]


To access original framework operation attribute value and connect to inputs, 
``node`` object of type ``NodeContext`` is used. It has two main methods:

* ``NodeContext::get_input`` to get input with a given index,

* ``NodeContext::get_attribute`` to get attribute value with a given name.

The conversion function should return a vector of node outputs that are mapped to 
corresponding outputs of the original framework operation in the same order.

@endsphinxdirective


