// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op_table.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
#define OP_CONVERTER(op) NamedOutputs op(const NodeContext& node)
OP_CONVERTER(argmax);
OP_CONVERTER(argmin);
OP_CONVERTER(assign);
OP_CONVERTER(assign_value);
OP_CONVERTER(batch_norm);
OP_CONVERTER(bicubic_interp_v2);
OP_CONVERTER(bilinear_interp_v2);
OP_CONVERTER(box_coder);
OP_CONVERTER(cast);
OP_CONVERTER(ceil);
OP_CONVERTER(clip);
OP_CONVERTER(concat);
OP_CONVERTER(conditional_block);
OP_CONVERTER(if_else_block);
OP_CONVERTER(conv2d);
OP_CONVERTER(conv2d_transpose);
OP_CONVERTER(cos);
OP_CONVERTER(cumsum);
OP_CONVERTER(deformable_conv);
OP_CONVERTER(dequantize_linear);
OP_CONVERTER(dropout);
OP_CONVERTER(elementwise_add);
OP_CONVERTER(elementwise_div);
OP_CONVERTER(elementwise_floordiv);
OP_CONVERTER(elementwise_max);
OP_CONVERTER(elementwise_min);
OP_CONVERTER(elementwise_mod);
OP_CONVERTER(elementwise_mul);
OP_CONVERTER(elementwise_pow);
OP_CONVERTER(elementwise_sub);
OP_CONVERTER(equal);
OP_CONVERTER(greater_equal);
OP_CONVERTER(not_equal);
OP_CONVERTER(elu);
OP_CONVERTER(embedding);
OP_CONVERTER(exp);
OP_CONVERTER(expand_v2);
OP_CONVERTER(expand_as_v2);
OP_CONVERTER(eye);
OP_CONVERTER(flip);
OP_CONVERTER(flatten_contiguous_range);
OP_CONVERTER(floor);
OP_CONVERTER(fill_any_like);
OP_CONVERTER(fill_constant);
OP_CONVERTER(fill_constant_batch_size_like);
OP_CONVERTER(gather);
OP_CONVERTER(gather_nd);
OP_CONVERTER(gelu);
OP_CONVERTER(greater_than);
OP_CONVERTER(grid_sampler);
OP_CONVERTER(group_norm);
OP_CONVERTER(hard_sigmoid);
OP_CONVERTER(hard_swish);
OP_CONVERTER(index_select);
OP_CONVERTER(layer_norm);
OP_CONVERTER(leaky_relu);
OP_CONVERTER(less_than);
OP_CONVERTER(less_equal);
OP_CONVERTER(linear_interp_v2);
OP_CONVERTER(linspace);
OP_CONVERTER(lod_array_length);
OP_CONVERTER(log);
OP_CONVERTER(logical_and);
OP_CONVERTER(logical_not);
OP_CONVERTER(logical_or);
OP_CONVERTER(logical_xor);
OP_CONVERTER(matmul);
OP_CONVERTER(matmul_v2);
OP_CONVERTER(matrix_nms);
OP_CONVERTER(meshgrid);
OP_CONVERTER(multiclass_nms);
OP_CONVERTER(nearest_interp_v2);
OP_CONVERTER(one_hot_v2);
OP_CONVERTER(p_norm);
OP_CONVERTER(pad3d);
OP_CONVERTER(partial_concat);
OP_CONVERTER(partial_sum);
OP_CONVERTER(pow);
OP_CONVERTER(pool2d);
OP_CONVERTER(pool3d);
OP_CONVERTER(pool3d_with_index);
OP_CONVERTER(prior_box);
OP_CONVERTER(quantize_linear);
OP_CONVERTER(range);
OP_CONVERTER(reduce_all);
OP_CONVERTER(reduce_max);
OP_CONVERTER(reduce_mean);
OP_CONVERTER(reduce_min);
OP_CONVERTER(reduce_prod);
OP_CONVERTER(reduce_sum);
OP_CONVERTER(relu);
OP_CONVERTER(relu6);
OP_CONVERTER(reshape2);
OP_CONVERTER(reverse);
OP_CONVERTER(rnn);
OP_CONVERTER(roi_align);
OP_CONVERTER(roll);
OP_CONVERTER(round);
OP_CONVERTER(rsqrt);
OP_CONVERTER(scale);
OP_CONVERTER(select_input);
OP_CONVERTER(set_value);
OP_CONVERTER(shape);
OP_CONVERTER(share_data);
OP_CONVERTER(sigmoid);
OP_CONVERTER(silu);
OP_CONVERTER(sin);
OP_CONVERTER(skip);
OP_CONVERTER(slice);
OP_CONVERTER(softmax);
OP_CONVERTER(softplus);
OP_CONVERTER(softshrink);
OP_CONVERTER(split);
OP_CONVERTER(sqrt);
OP_CONVERTER(squeeze);
OP_CONVERTER(stack);
OP_CONVERTER(strided_slice);
OP_CONVERTER(sum);
OP_CONVERTER(swish);
OP_CONVERTER(tanh);
OP_CONVERTER(tanh_shrink);
OP_CONVERTER(tensor_array_to_tensor);
OP_CONVERTER(tile);
OP_CONVERTER(top_k_v2);
OP_CONVERTER(transpose2);
OP_CONVERTER(tril_triu);
OP_CONVERTER(trilinear_interp_v2);
OP_CONVERTER(unsqueeze);
OP_CONVERTER(unique);
OP_CONVERTER(unstack);
OP_CONVERTER(where);
OP_CONVERTER(while_);
OP_CONVERTER(write_to_array);
OP_CONVERTER(where_index);
OP_CONVERTER(yolo_box);
OP_CONVERTER(generate_proposals_v2);
OP_CONVERTER(abs);
OP_CONVERTER(elu);
OP_CONVERTER(atan2);
OP_CONVERTER(scatter);
OP_CONVERTER(scatter_nd_add);
OP_CONVERTER(take_along_axis);
OP_CONVERTER(reduce_any);
}  // namespace op
std::map<std::string, CreatorFunction> get_supported_ops() {
    return {{"arg_max", op::argmax},
            {"arg_min", op::argmin},
            {"assign", op::assign},
            {"assign_value", op::assign_value},
            {"batch_norm", op::batch_norm},
            {"bicubic_interp_v2", op::bicubic_interp_v2},
            {"bilinear_interp_v2", op::bilinear_interp_v2},
            {"bilinear_interp", op::bilinear_interp_v2},
            {"bmm", op::matmul},
            {"box_coder", op::box_coder},
            {"cast", op::cast},
            {"ceil", op::ceil},
            {"clip", op::clip},
            {"concat", op::concat},
            {"conditional_block", op::conditional_block},
            {"conv2d", op::conv2d},
            {"conv2d_transpose", op::conv2d_transpose},
            {"cos", op::cos},
            {"cumsum", op::cumsum},
            {"deformable_conv", op::deformable_conv},
            {"deformable_conv_v1", op::deformable_conv},
            {"depthwise_conv2d", op::conv2d},
            {"depthwise_conv2d_transpose", op::conv2d_transpose},
            {"dequantize_linear", op::dequantize_linear},
            {"elementwise_add", op::elementwise_add},
            {"elementwise_div", op::elementwise_div},
            {"elementwise_floordiv", op::elementwise_floordiv},
            {"elementwise_mod", op::elementwise_mod},
            {"elementwise_mul", op::elementwise_mul},
            {"elementwise_max", op::elementwise_max},
            {"elementwise_min", op::elementwise_min},
            {"elementwise_sub", op::elementwise_sub},
            {"dropout", op::dropout},
            {"elementwise_pow", op::elementwise_pow},
            {"elu", op::elu},
            {"equal", op::equal},
            {"exp", op::exp},
            {"expand_v2", op::expand_v2},
            {"expand_as_v2", op::expand_as_v2},
            {"eye", op::eye},
            {"fill_any_like", op::fill_any_like},
            {"fill_constant", op::fill_constant},
            {"fill_constant_batch_size_like", op::fill_constant_batch_size_like},
            {"flatten_contiguous_range", op::flatten_contiguous_range},
            {"flatten", op::flatten_contiguous_range},
            {"flip", op::flip},
            {"floor", op::floor},
            {"gather", op::gather},
            {"gather_nd", op::gather_nd},
            {"gelu", op::gelu},
            {"generate_proposals_v2", op::generate_proposals_v2},
            {"greater_equal", op::greater_equal},
            {"greater_than", op::greater_than},
            {"grid_sampler", op::grid_sampler},
            {"group_norm", op::group_norm},
            {"hard_sigmoid", op::hard_sigmoid},
            {"hard_swish", op::hard_swish},
            {"index_select", op::index_select},
            {"layer_norm", op::layer_norm},
            {"leaky_relu", op::leaky_relu},
            {"less_than", op::less_than},
            {"less_equal", op::less_equal},
            {"linear_interp_v2", op::linear_interp_v2},
            {"linspace", op::linspace},
            {"lod_array_length", op::lod_array_length},
            {"log", op::log},
            {"logical_and", op::logical_and},
            {"logical_not", op::logical_not},
            {"logical_or", op::logical_or},
            {"logical_xor", op::logical_xor},
            {"lookup_table_v2", op::embedding},
            {"matmul", op::matmul},
            {"matmul_v2", op::matmul_v2},
            {"max_pool2d_with_index", op::pool2d},
            {"max_pool3d_with_index", op::pool3d_with_index},
            {"matrix_nms", op::matrix_nms},
            {"memcpy", op::skip},
            {"meshgrid", op::meshgrid},
            {"multiclass_nms3", op::multiclass_nms},
            {"nearest_interp_v2", op::nearest_interp_v2},
            {"nearest_interp", op::nearest_interp_v2},
            {"not_equal", op::not_equal},
            {"one_hot_v2", op::one_hot_v2},
            {"p_norm", op::p_norm},
            {"pad3d", op::pad3d},
            {"partial_concat", op::partial_concat},
            {"partial_sum", op::partial_sum},
            {"pow", op::pow},
            {"pool2d", op::pool2d},
            {"pool3d", op::pool3d},
            {"prior_box", op::prior_box},
            {"quantize_linear", op::quantize_linear},
            {"range", op::range},
            {"reduce_all", op::reduce_all},
            {"reduce_max", op::reduce_max},
            {"reduce_mean", op::reduce_mean},
            {"reduce_min", op::reduce_min},
            {"reduce_prod", op::reduce_prod},
            {"reduce_sum", op::reduce_sum},
            {"relu", op::relu},
            {"relu6", op::relu6},
            {"reshape2", op::reshape2},
            {"reverse", op::reverse},
            {"rnn", op::rnn},
            {"roi_align", op::roi_align},
            {"roll", op::roll},
            {"round", op::round},
            {"rsqrt", op::rsqrt},
            {"scale", op::scale},
            {"select_input", op::select_input},
            {"set_value", op::set_value},
            {"shape", op::shape},
            {"share_data", op::share_data},
            {"sigmoid", op::sigmoid},
            {"silu", op::silu},
            {"sin", op::sin},
            {"slice", op::slice},
            {"softmax", op::softmax},
            {"softplus", op::softplus},
            {"softshrink", op::softshrink},
            {"split", op::split},
            {"sqrt", op::sqrt},
            {"squeeze2", op::squeeze},
            {"stack", op::stack},
            {"strided_slice", op::strided_slice},
            {"sum", op::sum},
            {"swish", op::swish},
            {"sync_batch_norm", op::batch_norm},
            {"tanh", op::tanh},
            {"tanh_shrink", op::tanh_shrink},
            {"tensor_array_to_tensor", op::tensor_array_to_tensor},
            {"tile", op::tile},
            {"top_k_v2", op::top_k_v2},
            {"transpose2", op::transpose2},
            {"tril_triu", op::tril_triu},
            {"trilinear_interp_v2", op::trilinear_interp_v2},
            {"unsqueeze2", op::unsqueeze},
            {"unique", op::unique},
            {"unstack", op::unstack},
            {"where", op::where},
            {"while", op::while_},
            {"write_to_array", op::write_to_array},
            {"where_index", op::where_index},
            {"yolo_box", op::yolo_box},
            {"abs", op::abs},
            {"elu", op::elu},
            {"atan2", op::atan2},
            {"scatter", op::scatter},
            {"scatter_nd_add", op::scatter_nd_add},
            {"take_along_axis", op::take_along_axis},
            {"reduce_any", op::reduce_any},
            // paddle3.0
            {"batch_norm_", op::batch_norm},
            {"add", op::elementwise_add},
            {"argmax", op::argmax},
            {"multiply", op::elementwise_mul},
            {"reshape", op::reshape2},
            {"divide", op::elementwise_div},
            {"maximum", op::elementwise_max},
            {"minimum", op::elementwise_min},
            {"remainder", op::elementwise_mod},
            {"subtract", op::elementwise_sub},
            {"floor_divide", op::elementwise_floordiv},
            {"expand", op::expand_v2},
            {"assign_value_", op::assign_value},
            {"expand_as", op::expand_as_v2},
            {"full_like", op::fill_any_like},
            {"full_with_tensor", op::fill_constant},
            {"grid_sample", op::grid_sampler},
            {"hardsigmoid", op::hard_sigmoid},
            {"hardswish", op::hard_swish},
            {"one_hot", op::one_hot_v2},
            {"arange", op::range},
            {"all", op::reduce_all},
            {"max", op::reduce_max},
            {"mean", op::reduce_mean},
            {"min", op::reduce_min},
            {"prod", op::reduce_prod},
            // the type name sum already replaced by reduce_sum when decode json
            {"reduce_sum", op::reduce_sum},
            {"transpose", op::transpose2},
            {"set_value_with_tensor_", op::set_value},
            {"shape64", op::shape},
            {"share_data_", op::share_data},
            {"split_with_num", op::split},
            {"squeeze", op::squeeze},
            {"topk", op::top_k_v2},
            {"triu", op::tril_triu},
            {"tril", op::tril_triu},
            {"unsqueeze", op::unsqueeze},
            {"nonzero", op::where_index},
            {"any", op::reduce_any},
            {"add_n", op::sum},
            {"if", op::if_else_block},
            {"bicubic_interp", op::bicubic_interp_v2},
            {"embedding", op::embedding},
            {"generate_proposals", op::generate_proposals_v2},
            {"linear_interp", op::linear_interp_v2},
    };
};
const std::string& get_input_name_by_op_type(const std::string& type, size_t index) {
      static std::map<const std::string, const std::vector<std::string>> map = {
            {"argmax", {"X", "full"}},
            {"arg_min", {}},
            {"assign", {"X"}},
            {"assign_value", {}},
            {"assign_value_", {"X"}},
            {"batch_norm_", {"X", "Mean", "Variance", "Scale", "Bias"}},
            {"bicubic_interp_v2", {}},
            {"bicubic_interp", {"X", "SizeTensor"}},
            {"bilinear_interp_v2", {}},
            {"bilinear_interp", {"X", "SizeTensor"}},
            {"bmm", {"X", "Y"}},
            {"box_coder", {"PriorBox", "PriorBoxVar", "TargetBox"}},
            {"cast", {"X"}},
            {"ceil", {"X"}},
            {"clip", {"X", "min", "max"}},
            {"concat", {"X", "full"}},
            {"conditional_block", {}},
            {"if", {"Cond", "if_inputs", "else_inputs", "sub_block_indexs"}},
            {"conv2d", {"Input", "Filter"}},
            {"conv2d_transpose", {"Input", "Filter", "full"}},
            {"cos", {}},
            {"cumsum", {"X", "full"}},
            {"deformable_conv", {"Input", "Offset", "Filter", "Mask"}},
            {"deformable_conv_v1", {}},
            {"depthwise_conv2d", {"Input", "Filter"}},
            {"depthwise_conv2d_transpose", {"Input", "Filter", "full"}},
            {"dequantize_linear", {}},
            {"elementwise_add", {}},
            {"add", {"X", "Y"}},
            {"elementwise_div", {}},
            {"divide", {"X", "Y"}},
            {"elementwise_floordiv", {}},
            {"floor_divide", {"X", "Y"}},
            {"elementwise_mod", {}},
            {"remainder", {"X", "Y"}},
            {"elementwise_mul", {}},
            {"multiply", {"X", "Y"}},
            {"elementwise_max", {}},
            {"maximum", {"X", "Y"}},
            {"elementwise_min", {}},
            {"minimum", {"X", "Y"}},
            {"elementwise_sub", {}},
            {"subtract", {"X", "Y"}},
            {"dropout", {"X", "unused_input", "full"}},
            {"elementwise_pow", {"X", "Y"}},
            {"elu", {"X"}},
            {"equal", {"X", "Y"}},
            {"exp", {"X"}},
            {"expand_v2", {}},
            {"expand", {"X", "Shape"}},
            {"expand_as_v2", {}},
            {"expand_as", {"X", "Y"}},
            {"eye", {"num_rows", "num_columns"}},
            {"fill_any_like", {}},
            {"full_like", {"X", "full"}},
            {"fill_constant", {}},
            {"full_with_tensor", {"ValueTensor", "ShapeTensor"}},
            {"fill_constant_batch_size_like", {}},
            {"flatten_contiguous_range", {}},
            {"flatten", {"X"}},
            {"flip", {"X"}},
            {"floor", {"X"}},
            {"gather", {"X", "Index", "Axis"}},
            {"gather_nd", {"X", "Index"}},
            {"gelu", {"X"}},
            {"generate_proposals_v2", {}},
            {"generate_proposals", {"Scores", "BboxDeltas", "ImShape", "Anchors", "Variances"}},
            {"greater_equal", {"X", "Y"}},
            {"greater_than", {"X", "Y"}},
            {"grid_sample", {"X", "Grid"}},
            {"group_norm", {"X", "Scale", "Bias"}},
            {"hardsigmoid", {"X"}},
            {"hardswish", {"X"}},
            {"index_select", {"X", "Index"}},
            {"layer_norm", {}},
            {"leaky_relu", {"X"}},
            {"less_than", {"X", "Y"}},
            {"less_equal", {"X", "Y"}},
            {"linear_interp_v2", {}},
            {"linear_interp", {"X", "SizeTensor"}},
            {"linspace", {"Start", "Stop", "Num"}},
            {"lod_array_length", {}},
            {"log", {"X"}},
            {"logical_and", {"X", "Y"}},
            {"logical_not", {"X"}},
            {"logical_or", {"X", "Y"}},
            {"logical_xor", {"X", "Y"}},
            {"lookup_table_v2", {}},
            {"embedding", {"Ids", "W"}},
            {"matmul", {"X", "Y"}},
            {"matmul_v2", {}},
            {"max_pool2d_with_index", {"X"}},
            {"max_pool3d_with_index", {"X"}},
            {"matrix_nms", {"BBoxes", "Scores"}},
            {"memcpy", {}},
            {"meshgrid", {"X"}},
            {"multiclass_nms3", {"BBoxes", "Scores", "RoisNum"}},
            {"nearest_interp_v2", {}},
            {"nearest_interp", {"X", "SizeTensor"}},
            {"not_equal", {"X", "Y"}},
            {"one_hot_v2", {}},
            {"one_hot", {"X", "depth_tensor"}},
            {"p_norm", {"X"}},
            {"pad3d", {"X", "full"}},
            {"partial_concat", {}},
            {"partial_sum", {}},
            {"pow", {"X"}},
            {"pool2d", {"X", "ksize"}},
            {"pool3d", {"X"}},
            {"prior_box", {"Input", "Image"}},
            {"quantize_linear", {}},
            {"range", {}},
            {"arange", {"Start", "End", "Step"}},
            {"reduce_all", {}},
            {"all", {"X"}},
            {"reduce_max", {}},
            {"max", {"X", "full"}},
            {"reduce_mean", {}},
            {"mean", {"X", "full"}},
            {"reduce_min", {}},
            {"min", {"X", "full"}},
            {"reduce_prod", {}},
            {"prod", {"X", "full"}},
            // the type name sum already replaced by reduce_sum when decode json
            {"reduce_sum", {"X", "full"}},
            {"relu", {"X"}},
            {"relu6", {"X"}},
            {"reshape2", {}},
            {"reshape", {"X", "ShapeTensor"}},
            {"reverse", {}},
            {"rnn", {"Input", "PreState", "WeightList", "SequenceLength", "unused"}},
            {"roi_align", {}},
            {"roll", {"X", "ShiftsTensor"}},
            {"round", {"X"}},
            {"rsqrt", {}},
            {"scale", {"X", "ScaleTensor"}},
            {"select_input", {}},
            {"set_value", {}},
            {"set_value_with_tensor_", {"Input", "ValueTensor", "StartsTensorList", "EndsTensorList", "StepsTensorList"}},
            {"shape", {}},
            {"shape64", {"Input"}},
            {"share_data", {}},
            {"share_data_", {"X"}},
            {"sigmoid", {"X"}},
            {"silu", {"X"}},
            {"sin", {"X"}},
            {"slice", {"Input", "StartsTensor", "EndsTensor"}},
            {"softmax", {"X"}},
            {"softplus", {"X"}},
            {"softshrink", {"X"}},
            {"split", {}},
            {"split_with_num", {"X", "Input1", "Input2"}},
            {"sqrt", {"X"}},
            {"squeeze2", {}},
            {"squeeze", {"X", "full"}},
            {"stack", {"X"}},
            {"strided_slice", {"Input", "StartsTensor", "EndsTensor", "StridesTensor"}},
            {"sum", {}},
            {"add_n", {"X"}},
            {"swish", {"X"}},
            {"sync_batch_norm", {}},
            {"tanh", {"X"}},
            {"tanh_shrink", {"X"}},
            {"tensor_array_to_tensor", {}},
            {"tile", {}},
            {"top_k_v2", {}},
            {"topk", {"X", "K"}},
            {"transpose2", {}},
            {"transpose", {"X"}},
            {"tril_triu", {}},
            {"triu", {"X"}},
            {"tril", {"X"}},
            {"trilinear_interp_v2", {}},
            {"unsqueeze2", {}},
            {"unsqueeze", {"X", "AxesTensor"}},
            {"unique", {"X"}},
            {"unstack", {"X"}},
            {"where", {"Condition", "X", "Y"}},
            {"while", {}},
            {"write_to_array", {}},
            {"where_index", {}},
            {"nonzero", {"Condition"}},
            {"yolo_box", {"X", "ImgSize"}},
            {"abs", {"X"}},
            {"elu", {"Out"}},
            {"atan2", {"X1", "X2"}},
            {"scatter", {"X", "Ids", "Updates"}},
            {"scatter_nd_add", {"X", "Index", "Updates"}},
            {"take_along_axis", {}},
            {"reduce_any", {}},
            {"any", {"X", "full"}}
      };
      auto it = map.find(type);
      auto size = it->second.size();
      const static std::set<std::string> unknow_input_num_ops = {"sum"};
      auto unknow_it = unknow_input_num_ops.find(type);
      if (unknow_it != unknow_input_num_ops.end() && index >= size) {
          return it->second[size - 1];
      }
      bool success = (it != map.end() && (it->second.size() > index));
      FRONT_END_OP_CONVERSION_CHECK(success, "No input name found for ", type, " node.", " index:", index);
      return it->second[index];
}

const std::vector<std::string>& get_output_name_by_op_type(const std::string& type) {
      static std::map<const std::string, const std::vector<std::string>> map = {
            {"argmax", {"Out"}},
            {"arg_min", {}},
            {"assign", {"Out"}},
            {"assign_value", {}},
            {"assign_value_", {"Out"}},
            {"batch_norm_", {"Y"}},
            {"bicubic_interp_v2", {}},
            {"bicubic_interp", {"Out"}},
            {"bilinear_interp_v2", {}},
            {"bilinear_interp", {"Out"}},
            {"bmm", {"Out"}},
            {"box_coder", {"OutputBox"}},
            {"cast", {"Out"}},
            {"ceil", {"Out"}},
            {"clip", {"Out"}},
            {"concat", {"Out"}},
            {"conditional_block", {}},
            {"if", {"Out"}},
            {"conv2d", {"Output"}},
            {"conv2d_transpose", {"Output"}},
            {"cos", {}},
            {"cumsum", {"Out"}},
            {"deformable_conv", {"Output"}},
            {"deformable_conv_v1", {}},
            {"depthwise_conv2d", {"Output"}},
            {"depthwise_conv2d_transpose", {"Output"}},
            {"dequantize_linear", {}},
            {"elementwise_add", {}},
            {"add", {"Out"}},
            {"elementwise_div", {}},
            {"divide", {"Out"}},
            {"elementwise_floordiv", {}},
            {"floor_divide", {"Out"}},
            {"elementwise_mod", {}},
            {"remainder", {"Out"}},
            {"elementwise_mul", {}},
            {"multiply", {"Out"}},
            {"elementwise_max", {}},
            {"maximum", {"Out"}},
            {"elementwise_min", {}},
            {"minimum", {"Out"}},
            {"elementwise_sub", {}},
            {"subtract", {"Out"}},
            {"dropout", {"Out"}},
            {"elementwise_pow", {"Out"}},
            {"equal", {"Out"}},
            {"exp", {"Out"}},
            {"expand_v2", {}},
            {"expand", {"Out"}},
            {"expand_as_v2", {}},
            {"expand_as", {"Out"}},
            {"eye", {"Out"}},
            {"fill_any_like", {}},
            {"full_like", {"Out"}},
            {"fill_constant", {}},
            {"full_with_tensor", {"Out"}},
            {"fill_constant_batch_size_like", {}},
            {"flatten_contiguous_range", {}},
            {"flatten", {"Out"}},
            {"flip", {"Out"}},
            {"floor", {"Out"}},
            {"gather", {"Out"}},
            {"gather_nd", {"Out"}},
            {"gelu", {"Out"}},
            {"generate_proposals_v2", {}},
            {"generate_proposals", {"RpnRois", "RpnRoiProbs", "RpnRoisNum"}},
            {"greater_equal", {"Out"}},
            {"greater_than", {"Out"}},
            {"grid_sample", {"Output"}},
            {"group_norm", {"Y"}},
            {"hardsigmoid", {"Out"}},
            {"hardswish", {"Out"}},
            {"index_select", {"Out"}},
            {"layer_norm", {}},
            {"leaky_relu", {"Out"}},
            {"less_than", {"Out"}},
            {"less_equal", {"Out"}},
            {"linear_interp_v2", {}},
            {"linear_interp", {"Out"}},
            {"linspace", {"Out"}},
            {"lod_array_length", {}},
            {"log", {"Out"}},
            {"logical_and", {"Out"}},
            {"logical_not", {"Out"}},
            {"logical_or", {"Out"}},
            {"logical_xor", {"Out"}},
            {"lookup_table_v2", {}},
            {"embedding", {"Out"}},
            {"matmul", {"Out"}},
            {"matmul_v2", {}},
            {"max_pool2d_with_index", {"Out"}},
            {"max_pool3d_with_index", {"Out", "Mask"}},
            {"matrix_nms", {"Out", "Index", "RoisNum"}},
            {"memcpy", {}},
            {"meshgrid", {"Out"}},
            {"multiclass_nms3", {"Out", "Index", "NmsRoisNum"}},
            {"nearest_interp_v2", {}},
            {"nearest_interp", {"Out"}},
            {"not_equal", {"Out"}},
            {"one_hot_v2", {}},
            {"one_hot", {"Out"}},
            {"p_norm", {"Out"}},
            {"pad3d", {"Out"}},
            {"partial_concat", {}},
            {"partial_sum", {}},
            {"pow", {"Out"}},
            {"pool2d", {"Out"}},
            {"pool3d", {"Out"}},
            {"prior_box", {"Boxes", "Variances"}},
            {"quantize_linear", {}},
            {"range", {}},
            {"arange", {"Out"}},
            {"reduce_all", {}},
            {"all", {"Out"}},
            {"reduce_max", {}},
            {"max", {"Out"}},
            {"reduce_mean", {}},
            {"mean", {"Out"}},
            {"reduce_min", {}},
            {"min", {"Out"}},
            {"reduce_prod", {}},
            {"prod", {"Out"}},
            // the type name sum already replaced by reduce_sum when decode json
            {"reduce_sum", {"Out"}},
            {"relu", {"Out"}},
            {"relu6", {"Out"}},
            {"reshape2", {}},
            {"reshape", {"Out"}},
            {"reverse", {}},
            {"rnn", {"Out", "State"}},
            {"roi_align", {}},
            {"roll", {"Out"}},
            {"round", {"Out"}},
            {"rsqrt", {}},
            {"scale", {"Out"}},
            {"select_input", {}},
            {"set_value", {}},
            {"set_value_with_tensor_", {"Out"}},
            {"shape", {}},
            {"shape64", {"Out"}},
            {"share_data", {}},
            {"share_data_", {"Out"}},
            {"sigmoid", {"Out"}},
            {"silu", {"Out"}},
            {"sin", {"Out"}},
            {"slice", {"Out"}},
            {"softmax", {"Out"}},
            {"softplus", {"Out"}},
            {"softshrink", {"Out"}},
            {"split", {}},
            {"split_with_num", {"Out"}},
            {"sqrt", {"Out"}},
            {"squeeze2", {}},
            {"squeeze", {"Out"}},
            {"stack", {"Y"}},
            {"strided_slice", {"Out"}},
            {"sum", {}},
            {"add_n", {"Out"}},
            {"swish", {"Out"}},
            {"sync_batch_norm", {}},
            {"tanh", {"Out"}},
            {"tanh_shrink", {"Out"}},
            {"tensor_array_to_tensor", {}},
            {"tile", {}},
            {"top_k_v2", {}},
            {"topk", {"Out", "Indices"}},
            {"transpose2", {}},
            {"transpose", {"Out"}},
            {"tril_triu", {}},
            {"triu", {"Out"}},
            {"tril", {"Out"}},
            {"trilinear_interp_v2", {}},
            {"unsqueeze2", {}},
            {"unsqueeze", {"Out"}},
            {"unique", {"Out", "Index", "Inverse", "Counts"}},
            {"unstack", {"Y"}},
            {"where", {"Out"}},
            {"while", {}},
            {"write_to_array", {}},
            {"where_index", {}},
            {"nonzero", {"Out"}},
            {"yolo_box", {"Boxes", "Scores"}},
            {"abs", {"Out"}},
            {"elu", {"Out"}},
            {"atan2", {"Out"}},
            {"scatter", {"Out"}},
            {"scatter_nd_add", {"Out"}},
            {"take_along_axis", {}},
            {"reduce_any", {}},
            {"any", {"Out"}}
      };
      auto it = map.find(type);
      bool success = (it != map.end()) && (it->second.size() > 0);
      FRONT_END_OP_CONVERSION_CHECK(success, "No output name found for ", type, " node.");
      return it->second;
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
