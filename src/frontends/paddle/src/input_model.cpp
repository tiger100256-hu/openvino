// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"

#include <fstream>
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
#include "json_input_model_imp.hpp"
#include "proto_input_model_imp.hpp"

namespace ov {
namespace frontend {
namespace paddle {

using namespace ::paddle::framework::proto;
InputModel::InputModel(const std::string& path, const std::shared_ptr<TelemetryExtension>& telemetry) {
    std::string model_file{path};
    std::string ext = ".json";
    if (ov::util::ends_with(model_file, ext)) {
        _impl = std::make_shared<json::JsonInputModelImpl>(path, *this, telemetry);
    } else {
        _impl = std::make_shared<proto::ProtoInputModelImpl>(path, *this, telemetry);
    }
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
InputModel::InputModel(const std::wstring& path, const std::shared_ptr<TelemetryExtension>& telemetry) {
    std::string model_file{path};
    std::string ext = ".json";
    if (ov::util::ends_with(model_file, ext)) {
        _impl = std::make_shared<json::JsonInputModelImpl>(path, *this, telemetry);
    } else {
        _impl = std::make_shared<proto::ProtoInputModelImpl>(path, *this, telemetry);
    }
}
#endif

InputModel::InputModel(const std::vector<std::istream*>& streams, const std::shared_ptr<TelemetryExtension>& telemetry) {
    try {
        std::stringstream buffer;
        buffer << streams[0]->rdbuf();
        nlohmann::json data = nlohmann::json::parse(buffer);
        _impl = std::make_shared<json::JsonInputModelImpl>(streams, *this, telemetry);
    } catch (nlohmann::json::parse_error& e) {
        streams[0]->->seekg(0, streams[0]->beg);
        _impl = std::make_shared<proto::ProtoInputModelImpl>(streams, *this, telemetry);
    }
}

std::vector<std::shared_ptr<BaseOpPlace>> InputModel::get_op_places(const int32_t blck_idx) const {
    return _impl->get_op_places(blck_idx);
}

std::map<std::string, std::shared_ptr<BaseTensorPlace>> InputModel::get_var_places() const {
    return _impl->get_var_places();
}

std::map<paddle::TensorName, Output<Node>> InputModel::get_tensor_values() const {
    return _impl->get_tensor_values();
}

std::vector<Place::Ptr> InputModel::get_inputs() const {
    return _impl->get_inputs();
}

std::vector<Place::Ptr> InputModel::get_outputs() const {
    return _impl->get_outputs();
}

int64_t InputModel::get_version() const {
    return _impl->get_version();
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensorName) const {
    return _impl->get_place_by_tensor_name(tensorName);
}

Place::Ptr InputModel::get_place_by_input_index(size_t input_idx) const {
    FRONT_END_NOT_IMPLEMENTED(get_place_by_input_index);
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    _impl->override_all_outputs(outputs);
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    _impl->override_all_inputs(inputs);
}

void InputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    _impl->extract_subgraph(inputs, outputs);
}

void InputModel::set_partial_shape(const Place::Ptr& place, const ov::PartialShape& p_shape) {
    _impl->set_partial_shape(place, p_shape);
}

ov::PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    return _impl->get_partial_shape(place);
}

void InputModel::set_element_type(const Place::Ptr& place, const ov::element::Type& type) {
    _impl->set_element_type(place, type);
}


ov::element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    return castToTensorPlace(place)->get_element_type();
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    _impl->set_tensor_value(place, value);
}
std::shared_ptr<BaseTensorPlace> castToTensorPlace(const Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<BaseTensorPlace>(place)) {
        return var_place;
    } else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlace>(place)) {
        return in_port_place->get_source_tensor_paddle();
    } else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlace>(place)) {
        return out_port_place->get_target_tensor_paddle();
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlacepaddle.");
}

bool read_tensor(std::istream& is, char* data, size_t len) {
    is.read(data, len);
    return (size_t)is.gcount() == len;
}
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
