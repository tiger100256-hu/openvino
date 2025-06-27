// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "paddle_utils.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace frontend {
namespace paddle {

class BaseOpPlace;
class BaseTensorPlace;
class BaseInputModelImpl {
public:
    virtual std::vector<Place::Ptr> get_inputs() const = 0;
    virtual std::vector<Place::Ptr> get_outputs() const = 0;
    virtual int64_t get_version() const = 0;
    virtual Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const = 0;
    virtual void override_all_outputs(const std::vector<Place::Ptr>& outputs) = 0;
    virtual void override_all_inputs(const std::vector<Place::Ptr>& inputs) = 0;
    virtual void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) = 0;
    virtual void set_default_shape(Place::Ptr place, const ov::Shape&) = 0;
    virtual void set_partial_shape(Place::Ptr place, const ov::PartialShape&) = 0;
    virtual ov::PartialShape get_partial_shape(Place::Ptr place) const = 0;
    virtual void set_element_type(Place::Ptr place, const ov::element::Type&) = 0;
    virtual ov::element::Type get_element_type(const Place::Ptr& place) const = 0;
    virtual void set_tensor_value(Place::Ptr place, const void* value) = 0;
    virtual std::vector<std::shared_ptr<BaseOpPlace>> get_op_places(const int32_t blck_idx) const = 0;
    virtual std::map<std::string, std::shared_ptr<BaseTensorPlace>> get_var_places() const = 0;
    virtual std::map<paddle::TensorName, Output<Node>> get_tensor_values() const = 0;
};

bool read_tensor(std::istream& is, char* data, size_t len);
template <typename T>
std::basic_string<T> get_const_path(const std::basic_string<T>& folder_with_weights, const std::string& name) {
    return folder_with_weights + paddle::get_path_sep<T>() + name;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_const_path(const std::basic_string<wchar_t>& folder, const std::string& name) {
    return folder + paddle::get_path_sep<wchar_t>() + ov::util::string_to_wstring(name);
}
#endif

template <typename T>
bool is_pdmodel(const std::basic_string<T>& path) {
    std::string ext = ".pdmodel";
    return ov::util::ends_with(path, ext);
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
bool is_pdmodel(const std::basic_string<wchar_t>& path) {
    std::wstring ext = L".pdmodel";
    return ov::util::ends_with(path, ext);
}
#endif

template <typename T>
bool is_json_model(const std::basic_string<T>& path) {
    std::string ext = ".json";
    return ov::util::ends_with(path, ext);
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
bool is_json_model(const std::basic_string<wchar_t>& path) {
    std::wstring ext = L".json";
    return ov::util::ends_with(path, ext);
}
#endif

template <typename T>
std::basic_string<T> get_json_model_path(const std::basic_string<T>& path, std::ifstream* weights_stream) {
    std::string model_file{path};
    std::string ext = ".json";
    if (ov::util::ends_with(model_file, ext)) {
        std::string params_ext = ".pdiparams";
        std::string weights_file{path};
        weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
        weights_stream->open(weights_file, std::ios::binary);
        // Don't throw error if file isn't opened
        // It may mean that model don't have constants
    } else {
        model_file += paddle::get_path_sep<T>() + "__model__";
    }
    return model_file;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_json_model_path(const std::basic_string<wchar_t>& path, std::ifstream* weights_stream) {
    std::wstring model_file{path};
    std::wstring ext = L".json";
    if (ov::util::ends_with(model_file, ext)) {
        std::wstring params_ext = L".pdiparams";
        std::wstring weights_file{path};
        weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
        weights_stream->open(weights_file.c_str(), std::ios::binary);
        // Don't throw error if file isn't opened
        // It may mean that model don't have constants
    } else {
        model_file += paddle::get_path_sep<wchar_t>() + L"__model__";
    }
    return model_file;
}
#endif

template <typename T>
std::basic_string<T> get_model_path(const std::basic_string<T>& path, std::ifstream* weights_stream) {
    std::string model_file{path};
    std::string ext = ".pdmodel";
    if (ov::util::ends_with(model_file, ext)) {
        std::string params_ext = ".pdiparams";
        std::string weights_file{path};
        weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
        weights_stream->open(weights_file, std::ios::binary);
        // Don't throw error if file isn't opened
        // It may mean that model don't have constants
    } else {
        model_file += paddle::get_path_sep<T>() + "__model__";
    }
    return model_file;
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_model_path(const std::basic_string<wchar_t>& path, std::ifstream* weights_stream) {
    std::wstring model_file{path};
    std::wstring ext = L".pdmodel";
    if (ov::util::ends_with(model_file, ext)) {
        std::wstring params_ext = L".pdiparams";
        std::wstring weights_file{path};
        weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
        weights_stream->open(weights_file.c_str(), std::ios::binary);
        // Don't throw error if file isn't opened
        // It may mean that model don't have constants
    } else {
        model_file += paddle::get_path_sep<wchar_t>() + L"__model__";
    }
    return model_file;
}
#endif

class InputModel : public ov::frontend::InputModel {
public:
    explicit InputModel(const std::string& path, const std::shared_ptr<TelemetryExtension>& telemetry = {});
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    explicit InputModel(const std::wstring& path, const std::shared_ptr<TelemetryExtension>& telemetry = {});
#endif
    explicit InputModel(const std::vector<std::istream*>& streams,
                        const std::shared_ptr<TelemetryExtension>& telemetry = {});
    std::vector<Place::Ptr> get_inputs() const override;
    std::vector<Place::Ptr> get_outputs() const override;
    Place::Ptr get_place_by_tensor_name(const std::string& tensorName) const override;
    Place::Ptr get_place_by_input_index(size_t input_idx) const override;
    void override_all_outputs(const std::vector<Place::Ptr>& outputs) override;
    void override_all_inputs(const std::vector<Place::Ptr>& inputs) override;
    void extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) override;
    void set_partial_shape(const Place::Ptr& place, const ov::PartialShape&) override;
    ov::PartialShape get_partial_shape(const Place::Ptr& place) const override;
    void set_element_type(const Place::Ptr& place, const ov::element::Type&) override;
    ov::element::Type get_element_type(const Place::Ptr& place) const override;
    void set_tensor_value(const Place::Ptr& place, const void* value) override;
    int64_t get_version() const;

private:
    friend class ov::frontend::paddle::FrontEnd;
    std::shared_ptr<BaseInputModelImpl> _impl;

    std::vector<std::shared_ptr<BaseOpPlace>> get_op_places(const int32_t block_idx) const;
    std::map<std::string, std::shared_ptr<BaseTensorPlace>> get_var_places() const;
    std::map<std::string, Output<Node>> get_tensor_values() const;
};
std::shared_ptr<BaseTensorPlace> castToTensorPlace(const Place::Ptr& place);
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
