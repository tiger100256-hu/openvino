// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "input_model.hpp"
#include "place.hpp"
#include "framework.pb.h"
#include "paddle_utils.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "decoder_proto.hpp"
#include "openvino/opsets/opset7.hpp"
namespace ov {
namespace frontend {
namespace paddle {
namespace proto {

using namespace ::paddle::framework::proto;

class ProtoInputModelImpl : public BaseInputModelImpl {
public:
    template <typename T>
    ProtoInputModelImpl(const std::basic_string<T>& path,
                   const InputModel& input_model,
                   const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_fw_ptr{std::make_shared<ProgramDesc>()},
      m_input_model(input_model),
      m_telemetry(telemetry) {
          std::ifstream weights_stream;
          std::ifstream pb_stream(get_model_path<T>(path, &weights_stream).c_str(), std::ios::in | std::ifstream::binary);

          FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(),
                  "Could not open the file: \"",
                  util::path_to_string(path),
                  '"');
          FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(&pb_stream), "Model can't be parsed");
          // According to Paddle, the saved model has the framework version
          // For example Paddle 2.1.0 is encoded as 2001000. 0 means the latest framework.
          // https://github.com/paddle/Paddle/blob/develop/cmake/version.cmake
          // https://github.com/paddle/Paddle/blob/2100816c5190693cc7dee181e96af72e9f0fbd1d/paddle/fluid/framework/program_desc.cc#L52
          int64_t version = m_fw_ptr->version().version();
          FRONT_END_GENERAL_CHECK(
                  version >= 2000000 || version == 0,
                  "[Frontend]Only Support Paddle greater than 2.0.0, current version " + std::to_string(version));
          load_places();
          if (is_pdmodel(path)) {
              load_consts(&weights_stream);
          } else {
              load_consts(path);
          }
          create_temp_consts();
      }

    ProtoInputModelImpl(const std::vector<std::istream*>& streams,
                   const InputModel& input_model,
                   const std::shared_ptr<TelemetryExtension>& telemetry);
    std::vector<Place::Ptr> get_inputs() const;
    std::vector<Place::Ptr> get_outputs() const;
    int64_t get_version() const {
        return m_fw_ptr->version().version();
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
        void load_consts(const std::basic_string<T>& folder_with_weights) {
            for (const auto& item : m_var_places) {
                const auto& var_desc = std::dynamic_pointer_cast<ProtoTensorPlace>(item.second)->get_desc();
                const auto& name = item.first;
                if (ov::util::ends_with(name, std::string{"feed"}) || ov::util::ends_with(name, std::string{"fetch"}))
                    continue;
                if (!var_desc.persistable())
                    continue;

                FRONT_END_GENERAL_CHECK(var_desc.type().type() == ::paddle::framework::proto::VarType::LOD_TENSOR);
                const auto& tensor = var_desc.type().lod_tensor().tensor();
                Shape shape(tensor.dims().cbegin(), tensor.dims().cend());
                const auto& type = get_ov_type(tensor.data_type());
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
    std::shared_ptr<ProgramDesc> m_fw_ptr;
    const InputModel& m_input_model;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    std::map<paddle::TensorName, Output<Node>> m_tensor_values;

    std::shared_ptr<TelemetryExtension> m_telemetry;

    // shows if some nodes might be deleted from graph
    bool m_graph_changed = false;
};
}  // namespace proto
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
