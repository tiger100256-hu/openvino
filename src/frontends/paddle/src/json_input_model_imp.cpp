// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <fstream>
#include <memory>
#include <vector>
#if defined(__MINGW32__) || defined(__MINGW64__)
#    include <filesystem>
#endif
#include <queue>

#include "decoder_proto.hpp"
#include "framework.pb.h"
#include "input_model.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "paddle_utils.hpp"
#include "place.hpp"
#include "decoder_json.hpp"
#include "json_input_model_imp.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace json {
using namespace ::paddle::framework::proto;
void JsonInputModelImpl::load_places() {
    auto& graph = m_fw_ptr->m_graph;
    const int cnt_of_regions = graph.regions.size();
    std::map<std::string, uint64_t> op_statistics;
    uint32_t cnt_of_blocks = 0;
    for (int region_idx = 0; region_idx < cnt_of_regions; region_idx++) {
        auto& blocks = graph.regions[region_idx].blocks;
        cnt_of_blocks += blocks.size();
    }
    m_op_places.resize(cnt_of_blocks);
    uint32_t index = 0;
    for (int region_idx = 0; region_idx < cnt_of_regions; region_idx++) {
       auto& blocks = graph.regions[region_idx].blocks;
       for (int block_idx = 0; block_idx < blocks.size(); block_idx++, index++) {
           const auto& block = blocks[block_idx];
           for (const auto& op : block.ops) {
               auto op_place = std::make_shared<JsonOpPlace>(m_input_model, op);
               op_place->set_decoder(std::make_shared<DecoderJson>(op_place));
               if (m_telemetry) {
                   op_statistics[op.type]++;
               }
               m_op_places[index].push_back(op_place);
               for (const auto& output : op.outputPorts) {
                   auto out_port = std::make_shared<OutPortPlace>(m_input_model);
                   auto port_name = std::to_string(output.id);
                   m_var_places[port_name] = std::make_shared<JsonTensorPlace>(m_input_model, output);
                   if (op.is_parameter) {
                       m_const_name_to_id_map[op.name] = port_name;
                   }
                   if (op.type == "data") {
                       m_inputs.push_back(m_var_places[port_name]);
                   } else if (op.type == "fetch") {
                       m_outputs.push_back(m_var_places[port_name]);
                   }
                   // connect out_port and tensor
                   m_var_places[port_name]->add_producing_port(out_port);
                   out_port->set_target_tensor(m_var_places[port_name]);
                   // connect out_port and op
                   op_place->add_out_port(out_port, port_name);
                   out_port->set_op(op_place);
               }

               for (const auto& inputId : op.inputIds) {
                   auto in_port = std::make_shared<InPortPlace>(m_input_model);
                   auto port_name = std::to_string(inputId);

                   // connect in_port and tensor
                   const auto& tensor = m_var_places.at(port_name);
                   tensor->add_consuming_port(in_port);
                   in_port->set_source_tensor(tensor);

                   // connect in_port and op
                   op_place->add_in_port(in_port, port_name);
                   in_port->set_op(op_place);
               }
           }
        }
    }
    if (m_telemetry) {
        for (const auto& op : op_statistics) {
            m_telemetry->send_event("op_count", "paddle_" + op.first, static_cast<int>(op.second));
        }
    }
}

std::vector<std::shared_ptr<BaseOpPlace>> JsonInputModelImpl::get_op_places(const int32_t blck_idx) const {
    if (m_graph_changed) {
        return determine_cut_nodes();
    }
    if (static_cast<size_t>(blck_idx) < m_op_places.size())
        return m_op_places[blck_idx];
    return {};
}

std::vector<std::shared_ptr<BaseOpPlace>> JsonInputModelImpl::determine_cut_nodes() const {
    // std::queue<OpPlace*> q;
    // std::unordered_set<OpPlace*> visited;
    // std::vector<std::shared_ptr<JsonOpPlace>> new_op_places;
    // new_op_places.reserve(m_op_places[0].size());
    // // Marking nodes from outputs to inputs/constants
    // for (const auto& output : get_outputs()) {
    //     if (!output->is_input()) {
    //         auto paddle_output_op = std::dynamic_pointer_cast<JsonOpPlace>(output->get_producing_operation());
    //         FRONT_END_GENERAL_CHECK(paddle_output_op != nullptr, "Output doesn't have producing operation");
    //         if (!visited.count(paddle_output_op.get())) {
    //             visited.insert(paddle_output_op.get());
    //             q.push(paddle_output_op.get());
    //             new_op_places.push_back(paddle_output_op);
    //         }
    //     }
    // }
    // while (!q.empty()) {
    //     auto p_op = q.front();
    //     q.pop();
    //     for (const auto& map_pair : p_op->get_input_ports()) {
    //         for (const auto& port : map_pair.second) {
    //             auto tensor = port->get_source_tensor();
    //             if (tensor && !tensor->is_input() && !m_tensor_values.count(tensor->get_names()[0])) {
    //                 std::shared_ptr<JsonOpPlace> paddle_op =
    //                     std::dynamic_pointer_cast<JsonOpPlace>(tensor->get_producing_operation());
    //                 if (paddle_op && !visited.count(paddle_op.get())) {
    //                     visited.insert(paddle_op.get());
    //                     q.push(paddle_op.get());
    //                     new_op_places.push_back(paddle_op);
    //                 }
    //             }
    //         }
    //     }
    // }
    // std::reverse(new_op_places.begin(), new_op_places.end());
    // return new_op_places;
    return {};
}

// load_consts with stream is compatible with new PaddlePaddle API.
void JsonInputModelImpl::load_consts(std::istream* weight_stream) {
    std::set<std::string> param_names_set;
    for (const auto& block_op_places : m_op_places) {
        for (const auto& op_place : block_op_places) {
            const auto& op = std::dynamic_pointer_cast<JsonOpPlace>(op_place)->get_op();
            const auto& name = op.name;
            //if (ov::util::ends_with(name, std::string{"data"}) || ov::util::ends_with(name, std::string{"fetch"}))
            //    continue;

            // var_desc.persistable() is used to mark node const value or not.
            if (!op.is_parameter)
               continue;
            std::cout << "op.name:" << op.name << "op.type:" << op.type << std::endl;
            param_names_set.insert(name);
        }
    }
    // FRONT_END_GENERAL_CHECK(var_desc.type().type() == ::paddle::framework::proto::VarType::LOD_TENSOR);
    FRONT_END_GENERAL_CHECK(weight_stream != nullptr&& weight_stream->peek() != EOF,
                            "PaddlePaddle *.pdiparams format weight file doesn't exist!");
    for (auto& name : param_names_set) {
        /*
            reference:
            https://github.com/PaddlePaddle/Paddle2ONNX/blob/c14446437041a0aa3572994d085b7a35c5b0985c/paddle2onnx/parser/parser.cc#L261
            When deserialize the proto, the header of each weight
            [ 4 byte ]      -- version(not need)
            [   8 byte   ]  -- lod_level(not need)
            [ 4 byte ]      -- version(not need)
            [ 4 byte ]      -- TensorDesc size
            [ x byte ... ]  -- TensorDesc
            [ y byte ... ]  -- weight
        */
        {
            const size_t header_size = 16;
            std::vector<char> header(header_size);
            weight_stream->read(&header[0], header_size);
        }

        int32_t size;
        weight_stream->read(reinterpret_cast<char*>(&size), sizeof(size));

        std::unique_ptr<char[]> buf(new char[size]);
        weight_stream->read(reinterpret_cast<char*>(buf.get()), size);

        std::unique_ptr<::paddle::framework::proto::VarType_TensorDesc> tensor_desc(
            new ::paddle::framework::proto::VarType_TensorDesc());
        tensor_desc->ParseFromArray(buf.get(), size);
        Shape shape(tensor_desc->dims().cbegin(), tensor_desc->dims().cend());
        const auto& type = get_ov_type(tensor_desc->data_type());
        const auto& data_length = shape_size(shape) * type.size();
        std::cout << "name:" << name << " data_length:" << data_length << std::endl;
        std::vector<uint8_t> tensor_data(data_length);

        bool read_succeed = read_tensor(*weight_stream, reinterpret_cast<char*>(&tensor_data[0]), data_length);
        FRONT_END_GENERAL_CHECK(read_succeed,
                                "File containing constant with name ",
                                name,
                                " wasn't successfully read.");

        auto const_node = opset7::Constant::create(type, shape, &tensor_data[0]);
        const_node->set_friendly_name(name);
        m_tensor_values[m_const_name_to_id_map[name]] = const_node;
    }
}


void JsonInputModelImpl::create_temp_consts() {
    /*
    for (const auto& item : m_var_places) {
        const auto& var_place = item.second;
        const auto& var_desc = var_place->get_desc();
        const auto& name = item.first;
        if (var_desc.persistable())
            continue;

        // The node with tensorarray as its input may be created before the node with this tensorarray
        // as its output. e.g. the tensorarray is both the input and output of the same node.
        // So we have to create a fake empty node here.
        // Problem is, we have no idea which axis should be 0.
        // Since the models (faster/mask rcnn) are either concating tensors in tensorarray along the dynamic
        // dimension, or concating static shape tensors. So we make the dynamic dimension to be 0. In case of static
        // shape, we simply the the first dimension be 0.
        if (var_desc.type().has_tensor_array()) {
            const auto& tensor = var_desc.type().tensor_array().tensor();
            const auto& type = get_ov_type(tensor.data_type());

            std::cout << "WARNING: The PaddlePaddle model has \"TENSOR_ARRAY\" variables, which is supported "
                      << " under limited situations.\n";

            PartialShape tensor_ps(std::vector<Dimension>(tensor.dims().cbegin(), tensor.dims().cend()));
            tensor_ps.insert(tensor_ps.begin(), 1);  // unsqueeze
            // also update the place for following initialize the graph connection
            var_place->set_element_type(type);
            var_place->set_partial_shape(tensor_ps);

            Shape shape(tensor_ps.size(), 0);
            for (size_t i = 0; i < tensor_ps.size(); i++) {
                const auto& dim = tensor_ps[i];
                if (dim.is_static()) {
                    shape[i] = dim.get_length();
                }
            }

            if (tensor_ps.is_static()) {
                // this tensorarray tensor originally could be scalar, then
                // tensor_ps size would be 1 after unsqueeze.
                auto idx = tensor_ps.size() > 1 ? 1 : 0;
                shape[idx] = 0;
            }

            auto node = opset7::Constant::create(type, shape, {0});
            node->set_friendly_name(name);
            node->output(0).get_tensor().add_names({name});

            m_tensor_values[name] = node;
        }
    }
    */
}


JsonInputModelImpl::JsonInputModelImpl(const std::vector<std::istream*>& streams,
                                           const InputModel& input_model,
                                           const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_fw_ptr{std::make_shared<JsonProgramDesc>()},
      m_input_model(input_model),
      m_telemetry(telemetry) {
    if (streams.size() != 1) {
        FRONT_END_GENERAL_CHECK(streams.size() == 2,
                                "Two streams are needed to load a model: model and weights streams");
    }
    FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(*(streams[0])), "Model can't be parsed");
    load_places();
    if (streams.size() > 1)
        load_consts(streams[1]);
    create_temp_consts();
}

std::vector<Place::Ptr> JsonInputModelImpl::get_inputs() const {
    return m_inputs;
}

std::vector<Place::Ptr> JsonInputModelImpl::get_outputs() const {
    return m_outputs;
}

Place::Ptr JsonInputModelImpl::get_place_by_tensor_name(const std::string& tensorName) const {
    if (m_var_places.count(tensorName))
        return m_var_places.at(tensorName);
    return nullptr;
}

namespace {
std::shared_ptr<BaseTensorPlace> castToTensorPlace(const Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<JsonTensorPlace>(place)) {
        return var_place;
    } else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlace>(place)) {
        return in_port_place->get_source_tensor_paddle();
    } else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlace>(place)) {
        return out_port_place->get_target_tensor_paddle();
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlacepaddle.");
}

}  // namespace

void JsonInputModelImpl::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    m_graph_changed = true;
    m_inputs.clear();
    for (const auto& inp : inputs) {
        m_inputs.push_back(castToTensorPlace(inp));
    }
}

void JsonInputModelImpl::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    m_graph_changed = true;
    m_outputs.clear();
    for (const auto& outp : outputs) {
        m_outputs.push_back(castToTensorPlace(outp));
    }
}

void JsonInputModelImpl::extract_subgraph(const std::vector<Place::Ptr>& inputs,
                                                  const std::vector<Place::Ptr>& outputs) {
    m_graph_changed = true;
    override_all_inputs(inputs);
    override_all_outputs(outputs);
}

void JsonInputModelImpl::set_default_shape(Place::Ptr place, const ov::Shape& shape) {
    FRONT_END_NOT_IMPLEMENTED("set_default_shape");
}

void JsonInputModelImpl::set_partial_shape(Place::Ptr place, const ov::PartialShape& p_shape) {
    castToTensorPlace(place)->set_partial_shape(p_shape);
}

ov::PartialShape JsonInputModelImpl::get_partial_shape(Place::Ptr place) const {
    return castToTensorPlace(place)->get_partial_shape();
}

void JsonInputModelImpl::set_element_type(Place::Ptr place, const ov::element::Type& type) {
    castToTensorPlace(place)->set_element_type(type);
}

ov::element::Type JsonInputModelImpl::get_element_type(const Place::Ptr& place) const {
    return castToTensorPlace(place)->get_element_type();
}

void JsonInputModelImpl::set_tensor_value(Place::Ptr place, const void* value) {
    m_graph_changed = true;
    auto tensor_place = castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    auto constant = opset7::Constant::create(type, p_shape.to_shape(), value);
    auto name = tensor_place->get_names()[0];
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}
}  // namespace json
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
