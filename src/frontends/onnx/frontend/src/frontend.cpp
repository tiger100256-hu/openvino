// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <input_model.hpp>
#include <onnx_import/onnx.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/frontend/onnx/frontend.hpp>
#include <sstream>
#include <utils/onnx_internal.hpp>

#include "onnx_common/onnx_model_validator.hpp"

using namespace ov;
using namespace ov::frontend::onnx;

ONNX_FRONTEND_C_API ov::frontend::FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

ONNX_FRONTEND_C_API void* GetFrontEndData() {
    ov::frontend::FrontEndPluginInfo* res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "onnx";
    res->m_creator = []() {
        return std::make_shared<FrontEnd>();
    };
    return res;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.empty()) {
        return nullptr;
    }
    if (variants[0].is<std::string>()) {
        const auto path = variants[0].as<std::string>();
        return std::make_shared<InputModel>(path, m_extensions);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    if (variants[0].is<std::wstring>()) {
        const auto path = variants[0].as<std::wstring>();
        return std::make_shared<InputModel>(path, m_extensions);
    }
#endif
    if (variants[0].is<std::istream*>()) {
        const auto stream = variants[0].as<std::istream*>();
        if (variants.size() > 1 && variants[1].is<std::string>()) {
            const auto path = variants[0].as<std::string>();
            return std::make_shared<InputModel>(*stream, path, m_extensions);
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        if (variants.size() > 1 && variants[1].is<std::wstring>()) {
            const auto path = variants[1].as<std::wstring>();
            return std::make_shared<InputModel>(*stream, path, m_extensions);
        }
#endif
        return std::make_shared<InputModel>(*stream, m_extensions);
    }
    return nullptr;
}

std::shared_ptr<ngraph::Function> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto model_onnx = std::dynamic_pointer_cast<InputModel>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->convert();
}

void FrontEnd::convert(const std::shared_ptr<ov::Model>& partially_converted) const {
    ngraph::onnx_import::detail::convert_decoded_function(partially_converted);
}

std::shared_ptr<ngraph::Function> FrontEnd::decode(const InputModel::Ptr& model) const {
    auto model_onnx = std::dynamic_pointer_cast<InputModel>(model);
    NGRAPH_CHECK(model_onnx != nullptr, "Invalid input model");
    return model_onnx->decode();
}

std::string FrontEnd::get_name() const {
    return "onnx";
}

namespace {
/**
 * This helper struct uses RAII to rewind/reset the stream so that it points to the beginning
 * of the underlying resource (string, file, and so on). It works similarly to std::lock_guard,
 * which releases a mutex upon destruction.
 *
 * This ensures that the stream is always reset (exception, successful and unsuccessful
 * model validation).
 */
struct StreamRewinder {
    StreamRewinder(std::istream& stream) : m_stream(stream) {
        m_stream.seekg(0, m_stream.beg);
    }
    ~StreamRewinder() {
        m_stream.seekg(0, m_stream.beg);
    }

private:
    std::istream& m_stream;
};
}  // namespace

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 0) {
        return false;
    }
    std::ifstream model_stream;
    if (variants[0].is<std::string>()) {
        const auto path = variants[0].as<std::string>();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        const auto path = variants[0].as<std::wstring>();
        model_stream.open(path, std::ios::in | std::ifstream::binary);
    }
#endif
    if (model_stream.is_open()) {
        model_stream.seekg(0, model_stream.beg);
        const bool is_valid_model = ngraph::onnx_common::is_valid_model(model_stream);
        model_stream.close();
        return is_valid_model;
    }
    if (variants[0].is<std::istream*>()) {
        const auto stream = variants[0].as<std::istream*>();
        StreamRewinder rwd{*stream};
        return ngraph::onnx_common::is_valid_model(*stream);
    }

    return false;
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_extensions.telemetry = telemetry;
    } else if (auto progress_reporter = std::dynamic_pointer_cast<ProgressReporterExtension>(extension)) {
        m_extensions.progress_reporter = progress_reporter;
    }
}
