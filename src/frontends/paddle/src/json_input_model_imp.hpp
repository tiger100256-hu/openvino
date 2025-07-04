// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "input_model.hpp"
#include "json_data.hpp"
#include "paddle_utils.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"


namespace ov {
namespace frontend {
namespace paddle {
namespace json {
class JsonProgramDesc {
public:
    bool ParseFromIstream (std::istream& pb_stream) {
        try {
            m_graph.json_data = nlohmann::json::parse(pb_stream);
        } catch (nlohmann::json::parse_error& e) {
            return false;
        }
        auto& base_code = m_graph.json_data.at("base_code");
        m_graph.magic = base_code.at("magic").template get<std::string>();
        m_graph.trainable = base_code.at("trainable").template get<bool>();
        m_graph.version = base_code.at("version").template get<uint64_t>();
        auto& programJson = m_graph.json_data.at("program");
        auto& regionsJson = programJson.at("regions");
        for (auto& regionJson : regionsJson) {
            auto region = std::make_shared<Region>()
            json::decodeRegion(regionJson, region);
            m_graph.regions.push_back(std::move(region));
        }
        return true;
    }

    // use for parse sub block in if node
    bool ParseFromJson(const nlohmann::json& sub_json) {
        m_graph.magic = 'sub_pir'
        m_graph.trainable = 'false';
        m_graph.version = '1'
        auto& regionsJson = sub_json.at("regions");
        for (auto& regionJson : regionsJson) {
            Region newRegion;
            json::decodeRegion(regionJson, newRegion);
            m_graph.regions.push_back(std::move(newRegion));
        }
        return true;
    }

    int64_t version();
    Graph m_graph;
};

class JsonInputModelImpl : public BaseInputModelImpl {
public:
    template <typename T>
    JsonInputModelImpl(const std::basic_string<T>& path,
                   const InputModel& input_model,
                   const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_fw_ptr{std::make_shared<JsonProgramDesc>()},
      m_input_model(input_model),
      m_telemetry(telemetry) {
    std::ifstream weights_stream;
    std::ifstream pb_stream(get_json_model_path<T>(path, &weights_stream).c_str(), std::ios::in | std::ifstream::binary);

    FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(),
                            "Could not open the file: \"",
                            util::path_to_string(path),
                            '"');
    FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(pb_stream), "Model can't be parsed");
    load_places();
    if (is_json_model(path)) {
        load_consts(&weights_stream);
    } else {
        load_consts(path);
    }
    create_temp_consts();
    }

    JsonInputModelImpl(const std::vector<std::istream*>& streams,
                   const InputModel& input_model,
                   const std::shared_ptr<TelemetryExtension>& telemetry);

    std::vector<Place::Ptr> get_inputs() const;
    std::vector<Place::Ptr> get_outputs() const;
    int64_t get_version() const {
        return m_fw_ptr->m_graph.version;
    }
    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const;
    void override_all_outputs(const std::vector<Place::Ptr>& outputs);
    void override_all_inputs(const std::vector<Place::Ptr>& inputs);
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs);
    void set_default_shape(Place::Ptr place, const ov::Shape&);
    void set_partial_shape(Place::Ptr place, const ov::PartialShape&);
    ov::PartialShape get_partial_shape(Place::Ptr place) const;
    void set_element_type(Place::Ptr place, const ov::element::Type&);
    ov::element::Type get_element_type(const Place::Ptr& place) const;
    void set_tensor_value(Place::Ptr place, const void* value);
    std::vector<std::shared_ptr<BaseOpPlace>> get_op_places(const int32_t blck_idx) const;
    std::map<std::string, std::shared_ptr<BaseTensorPlace>> get_var_places() const {
        return m_var_places;
    }
    std::map<paddle::TensorName, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    };

private:
    void load_places();
    template <typename T>
    void load_consts(const std::basic_string<T>& folder_with_weights){
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
           for (auto& name : param_names_set) {
               if (!folder_with_weights.empty()) {
               #if defined(__MINGW32__) || defined(__MINGW64__)
                   std::ifstream is(std::filesystem::path(get_const_path(folder_with_weights, name)),
                                    std::ios::in | std::ifstream::binary);
               #else
                   std::ifstream is(get_const_path(folder_with_weights, name), std::ios::in | std::ifstream::binary);
               #endif
                   FRONT_END_GENERAL_CHECK(is && is.is_open(), "Cannot open file for constant value.");
               auto* weight_stream = &is; // FRONT_END_GENERAL_CHECK(var_desc.type().type() == ::paddle::framework::proto::VarType::LOD_TENSOR);
               FRONT_END_GENERAL_CHECK(weight_stream != nullptr&& weight_stream->peek() != EOF,
                                       "PaddlePaddle *.pdiparams format weight file doesn't exist!");
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
               // std::cout << "name:" << name << " data_length:" << data_length << std::endl;
               std::vector<uint8_t> tensor_data(data_length);

               bool read_succeed = read_tensor(*weight_stream, reinterpret_cast<char*>(&tensor_data[0]), data_length);
               FRONT_END_GENERAL_CHECK(read_succeed,
                                       "File containing constant with name ",
                                       name,
                                       " wasn't successfully read.");

               auto const_node = opset7::Constant::create(type, shape, &tensor_data[0]);
               // if (shape_size(shape) > 8 * 2) {
               //     auto* data = (float*)(&tensor_data[0]);
               //     float a  = *data;
               //     float b  = *(data + 1);
               //     std::cout << " "  << a << " " << b << std::endl;
               // }
               const_node->set_friendly_name(name);
               m_tensor_values[m_const_name_to_id_map[name]] = const_node;
            }
        }
    }
    void load_consts(std::istream* weight_stream);
    void create_temp_consts();
    std::vector<std::shared_ptr<BaseOpPlace>> determine_cut_nodes() const;

    std::vector<std::vector<std::shared_ptr<BaseOpPlace>>> m_op_places;
    std::map<std::string, std::shared_ptr<BaseTensorPlace>> m_var_places;
    std::map<std::string, std::string> m_const_name_to_id_map;
    std::shared_ptr<JsonProgramDesc> m_fw_ptr;
    const InputModel& m_input_model;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    std::map<paddle::TensorName, Output<Node>> m_tensor_values;

    std::shared_ptr<TelemetryExtension> m_telemetry;

    // shows if some nodes might be deleted from graph
    bool m_graph_changed = false;
};
}  // namespace json
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
