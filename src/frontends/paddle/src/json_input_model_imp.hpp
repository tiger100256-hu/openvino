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
    load_consts(&weights_stream);
    //create_temp_consts();
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
            for (const auto& item : m_var_places) {
                const auto& port = std::dynamic_pointer_cast<JsonTensorPlace>(item.second)->get_port();
                const auto& name = item.first;
                if (ov::util::ends_with(name, std::string{"data"}) || ov::util::ends_with(name, std::string{"fetch"}))
                    continue;
                //if (!var_desc.persistable())
                //    continue;

                // FRONT_END_GENERAL_CHECK(var_desc.type().type() == ::paddle::framework::proto::VarType::LOD_TENSOR);
                // const auto& tensor = var_desc.type().lod_tensor().tensor();
                Shape shape(port.shapes);
                const auto& type = convert_to_ov_type(port.precision);
                const auto& data_length = shape_size(shape) * type.size();
                std::vector<uint8_t> tensor_data(data_length);

                bool read_succeed = false;
                if (!folder_with_weights.empty()) {
#if defined(__MINGW32__) || defined(__MINGW64__)
                    std::ifstream is(std::filesystem::path(get_const_path(folder_with_weights, name)),
                            std::ios::in | std::ifstream::binary);
#else
                    std::ifstream is(get_const_path(folder_with_weights, name), std::ios::in | std::ifstream::binary);
#endif
                    FRONT_END_GENERAL_CHECK(is && is.is_open(), "Cannot open file for constant value.");
                    const size_t header_size = 16;
                    std::vector<char> header(header_size);
                    is.read(&header[0], header_size);

                    uint32_t dims_len = 0;
                    is.read(reinterpret_cast<char*>(&dims_len), 4);
                    std::vector<char> dims_struct(dims_len);
                    is.read(&dims_struct[0], dims_len);
                    read_succeed = read_tensor(is, reinterpret_cast<char*>(&tensor_data[0]), data_length);
                } else {
                    FRONT_END_GENERAL_CHECK(false, "Folder with weights must be provided.");
                }
                FRONT_END_GENERAL_CHECK(read_succeed,
                        "File containing constant with name ",
                        name,
                        " wasn't successfully read.");
                auto const_node = opset7::Constant::create(type, shape, &tensor_data[0]);
                const_node->set_friendly_name(name);
                m_tensor_values[name] = const_node;
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
