// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder_json.hpp"
#include "op_table.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>


namespace ov {
namespace frontend {
namespace paddle {

using namespace ::paddle::framework;

ov::Any DecoderJson::get_attribute(const std::string& name) const {
    static const std::map<std::string, std::map<std::string, std::string>> attr_name_map = {
        {"batch_norm_", {{"data_layout", "data_format"}}},
        {"cast", {{"out_dtype", "dtype"}}},
        {"pool3d", {{"ksize", "kernel_size"}}}
        };
    auto& op = op_place.lock()->get_op();
    std::string new_name = name;
    auto op_it = attr_name_map.find(op.type);
    if (op_it != attr_name_map.end()) {
        const auto attr_names = op_it->second;
        auto at_it = attr_names.find(name);
        if (at_it != attr_names.end()) {
            new_name = at_it->second;
        }
    }
    auto& attrs = op.json_data.at("A");
    for (auto& attr : attrs) {
        std::string attr_name = attr.at("N").template get<std::string>();
        if (attr_name == new_name) {
            return json::decode_attr(attr);
        }
    }
    return {};
}

int64_t DecoderJson::get_version() const {
    return 1;
}

ov::Any DecoderJson::convert_attribute(const Any& data, const std::type_info& type_info) const {
    // if (data.is<int32_t>() && type_info == typeid(ov::element::Type)) {
    //     return get_ov_type(static_cast<proto::VarType_Type>(data.as<int32_t>()));
    // } else if (data.is<std::vector<int32_t>>() && type_info == typeid(std::vector<ov::element::Type>)) {
    //     const auto& casted = data.as<std::vector<int32_t>>();
    //     std::vector<ov::element::Type> types(casted.size());
    //     for (size_t i = 0; i < casted.size(); ++i) {
    //         types[i] = get_ov_type(static_cast<proto::VarType_Type>(casted[i]));
    //     }
    //     return types;
    // }
    // no conversion rules found.
    return data;
}

std::vector<paddle::OutPortName> DecoderJson::get_output_names() const {
    std::vector<std::string> output_names;
    auto& op = op_place.lock()->get_op();
    auto fix_output_name = get_output_name_by_op_type(op.type);
    for (auto& name : fix_output_name) {
        output_names.push_back(name);
    }

    return output_names;
}
// ?
std::vector<paddle::TensorName> DecoderJson::get_output_var_names(const std::string& var_name) const {
    // std::vector<std::string> output_names;
    // for (const auto& output : get_place()->get_desc().outputs()) {
    //     if (output.parameter() == var_name) {
    //         for (int idx = 0; idx < output.arguments_size(); ++idx) {
    //             output_names.push_back(output.arguments()[idx]);
    //         }
    //     }
    // }
    // return output_names;
    return {var_name};
}
// ?
std::vector<paddle::TensorName> DecoderJson::get_input_var_names(const std::string& var_name) const {
    // std::vector<std::string> input_names;
    // for (const auto& input : get_place()->get_desc().inputs()) {
    //     if (input.parameter() == var_name) {
    //         for (int idx = 0; idx < input.arguments_size(); ++idx) {
    //             input_names.push_back(input.arguments()[idx]);
    //         }
    //     }
    // }
    // return input_names;
    return {var_name};
}

size_t DecoderJson::get_output_size(const std::string& port_name) const {
    // const auto out_port = get_place()->get_output_ports().at(port_name);
    // return out_port.size();
    return 1;
}

size_t DecoderJson::get_output_size() const {
    auto& op = get_place()->get_op();
    return op.outputPorts.size();
}

std::map<std::string, std::vector<ov::element::Type>> DecoderJson::get_output_type_map() const {
    auto& op = get_place()->get_op();
    std::map<std::string, std::vector<ov::element::Type>> output_types;
    for (const auto& outputport : op.outputPorts) {
        output_types[std::to_string(outputport.id)].push_back(json::convert_to_ov_type(outputport.precision));
    }
    return output_types;
}

std::vector<std::pair<ov::element::Type, ov::PartialShape>> DecoderJson::get_output_port_infos(
    const std::string& port_name) const {
    auto& op = get_place()->get_op();
    auto fix_output_name = get_output_name_by_op_type(op.type);
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> output_types;
    size_t index = 0;
    for (auto& output_name : fix_output_name) {
        if (output_name == port_name) {
            break;
        } else {
            index++;
        }
    }
    for (const auto& outputport : op.outputPorts) {
        output_types.push_back({json::convert_to_ov_type(outputport.precision),
                ov::PartialShape(outputport.shapes)});
    }
    if (index > output_types.size() -1 ) {
       FRONT_END_GENERAL_CHECK(false, "can't find ouput name ", port_name);
       return{};
    }
    return {output_types[index]};
}

ov::element::Type DecoderJson::get_out_port_type(const std::string& port_name) const {
    auto map = get_output_type_map();
    auto iter = map.find(port_name);
    if(iter != map.end()) {
       return iter->second[0];
    } else {
       FRONT_END_GENERAL_CHECK(false, "get port precision failed", port_name);
       return ov::element::undefined;
    }
}

std::string DecoderJson::get_op_type() const {
    return get_place()->get_op().type;
}
/*
std::vector<proto::OpDesc_Attr> DecoderJson::decode_attribute_helper(const std::string& name) const {
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto& attr : get_place()->get_desc().attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    FRONT_END_GENERAL_CHECK(attrs.size() <= 1,
                            "An error occurred while parsing the ",
                            name,
                            " attribute of ",
                            get_place()->get_desc().type(),
                            "node. Unsupported number of attributes. Current number: ",
                            attrs.size(),
                            " Expected number: 0 or 1");
    return attrs;
}

namespace {
inline std::map<std::string, OutputVector> map_for_each_input_impl(
    const google::protobuf::RepeatedPtrField<::paddle::framework::proto::OpDesc_Var>& c,
    const std::function<Output<Node>(const std::string&, size_t)>& func) {
    size_t idx = 0;
    std::map<std::string, OutputVector> res;
    for (const auto& port : c) {
        std::vector<Output<Node>> v;
        v.reserve(port.arguments_size());
        for (const auto& inp : port.arguments()) {
            v.push_back(func(inp, idx++));
        }
        res.emplace(std::make_pair(port.parameter(), v));
    }
    return res;
}
}  // namespace
*/

std::map<std::string, OutputVector> DecoderJson::map_for_each_input(
    const std::function<Output<Node>(const std::string&, size_t)>& func) const {
    // return map_for_each_input_impl(get_place()->get_desc().inputs(), func);
    FRONT_END_GENERAL_CHECK(false, "haven't implemented.");
    return {};
}

std::map<std::string, OutputVector> DecoderJson::map_for_each_output(
    const std::function<Output<Node>(const std::string&, size_t)>& func) const {
    // return map_for_each_input_impl(get_place()->get_desc().outputs(), func);
    FRONT_END_GENERAL_CHECK(false, "haven't implemented.");
    return {};
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
