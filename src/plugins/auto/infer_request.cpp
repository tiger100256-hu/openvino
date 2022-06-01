// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "infer_request.hpp"
#include <ie_input_info.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <blob_factory.hpp>

namespace MultiDevicePlugin {

using namespace InferenceEngine;

// ------------------------------MultiDeviceInferRequest----------------------------
MultiDeviceInferRequest::MultiDeviceInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                 const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                                 const InferenceEngine::SoIInferRequestInternal & request_to_share_blobs_with,
                                                 InferenceEngine::RemoteContext::Ptr ctx)
        : IInferRequestInternal(inputs, outputs),
          _sharedRequest(request_to_share_blobs_with)  {
    CreateInferRequest(request_to_share_blobs_with, ctx);
}

MultiDeviceInferRequest::MultiDeviceInferRequest(const InputsDataMap&   networkInputs,
                                                 const OutputsDataMap&  networkOutputs,
                                                 const SoIInferRequestInternal & request_to_share_blobs_with,
                                                 InferenceEngine::RemoteContext::Ptr ctx)
        : IInferRequestInternal(networkInputs, networkOutputs),
          _sharedRequest(request_to_share_blobs_with) {
    CreateInferRequest(request_to_share_blobs_with, ctx);
}

void MultiDeviceInferRequest::CreateInferRequest(const InferenceEngine::SoIInferRequestInternal& request_to_share_blobs_with,
            InferenceEngine::RemoteContext::Ptr ctx) {
    if (request_to_share_blobs_with) {
        // borrow device-friendly blobs from the request
        for (const auto &it : _networkInputs)
            _inputs[it.first] = request_to_share_blobs_with->GetBlob(it.first);
        for (const auto &it : _networkOutputs)
            _outputs[it.first] = request_to_share_blobs_with->GetBlob(it.first);
        return;
    }
    // Allocate all input blobs
    for (const auto &it : _networkInputs) {
        auto l = it.second->getLayout();
        auto p = it.second->getPrecision();
        auto dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        if (ctx) {
            _inputs[it.first] = ctx->CreateHostBlob(desc);
        } else {
            _inputs[it.first] = make_blob_with_precision(desc);
        }
        _inputs[it.first]->allocate();
    }
    // Allocate all output blobs
    for (const auto &it : _networkOutputs) {
        auto l = it.second->getLayout();
        auto p = it.second->getPrecision();
        auto dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        if (ctx) {
            _outputs[it.first] = ctx->CreateHostBlob(desc);
        } else {
            _outputs[it.first] = make_blob_with_precision(desc);
        }
        _outputs[it.first]->allocate();
    }
}
void MultiDeviceInferRequest::SetBlobsToAnotherRequest(const SoIInferRequestInternal& req) {
    for (const auto &it : _networkInputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
    for (const auto &it : _networkOutputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req->GetBlob(name) != blob)
            req->SetBlob(name, blob);
    }
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> MultiDeviceInferRequest::GetPerformanceCounts() const {
    return _perfMap;
}

void MultiDeviceInferRequest::InferImpl() {
    IE_THROW(NotImplemented);
}

}  // namespace MultiDevicePlugin
