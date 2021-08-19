// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>

#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include <ie_plugin_config.hpp>
#include "multi_device_exec_network.hpp"
#include "multi_device_async_infer_request.hpp"
#include "multi_device_plugin.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "transformations/utils/utils.hpp"

// ------------------------------MultiDeviceExecutableNetwork----------------------------
namespace MultiDevicePlugin {
using namespace InferenceEngine;

namespace {
std::string GetNetworkPrecision(const InferenceEngine::CNNNetwork &network) {
    auto nGraphFunc = network.getFunction();
    bool isINTModel = ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc);
    if (isINTModel) {
        return METRIC_VALUE(INT8);
    }
    for (auto & node : nGraphFunc->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node) ||
            std::dynamic_pointer_cast<ngraph::op::DeconvolutionIE>(node)) {
            auto layerType = node->input(1).get_element_type().get_type_name();
            if (layerType == "f32")
                return METRIC_VALUE(FP32);
            if (layerType == "f16")
                return METRIC_VALUE(FP16);
        }
    }
    return METRIC_VALUE(FP32);
}
}  // namespace

thread_local MultiDeviceExecutableNetwork::WorkerInferRequest* MultiDeviceExecutableNetwork::_thisWorkerInferRequest = nullptr;
// TODO: revert to the plain variable (see header file), when we moved to the next CentOS 8.x in our support matrix
thread_local const char* MultiDeviceExecutableNetwork::_thisPreferredDeviceName = "";

struct IdleGuard {
    explicit IdleGuard(MultiDeviceExecutableNetwork::WorkerInferRequest* workerInferRequestPtr,
                       MultiDeviceExecutableNetwork::NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->try_push(_workerInferRequestPtr);
        }
    }
    MultiDeviceExecutableNetwork::NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    MultiDeviceExecutableNetwork::WorkerInferRequest*     _workerInferRequestPtr = nullptr;
    MultiDeviceExecutableNetwork::NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};

MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork(const DeviceMap<InferenceEngine::SoExecutableNetworkInternal>&       networksPerDevice,
                                                           const std::vector<DeviceInformation>&                                networkDevices,
                                                           const std::unordered_map<std::string, InferenceEngine::Parameter>&   config,
                                                           const bool                                                           needPerfCounters) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _devicePriorities{networkDevices},
    _devicePrioritiesInitial{networkDevices},
    _networksPerDevice{networksPerDevice},
    _config{config},
    _needPerfCounters{needPerfCounters} {
    _taskExecutor.reset();
    for (auto&& networkValue : _networksPerDevice) {
        auto& device  = networkValue.first;
        auto& network = networkValue.second;
        GenerateWorkers(device, network);
    }
}

MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork(const std::string&                         modelPath,
                                                           const InferenceEngine::CNNNetwork&         network,
                                                           const std::map<std::string, std::string>&  config,
                                                           MultiDeviceInferencePlugin*                plugin)
                                                           : _multiPlugin(plugin)
                                                           , _workModeIsAUTO(true) {
    if (_multiPlugin->GetCore() == nullptr) {
        IE_THROW() << "Please, work with MULTI device via InferencEngine::Core object";
    }

    if (modelPath.empty() && network.getFunction() == nullptr) {
        IE_THROW() << "MULTI device supports just ngraph network representation";
    }

    auto strDevices = _multiPlugin->GetDeviceList(config);
    // collect the settings that are applicable to the devices we are loading the network to
    _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = strDevices;

    auto metaDevices = _multiPlugin->ParseMetaDevices(strDevices, config);
    _devicePrioritiesInitial = metaDevices;
    _devicePriorities = metaDevices;

    // init worker queue
    for (auto&& device : metaDevices) {
        _idleWorkerRequests[device.deviceName];
    }

    auto core = _multiPlugin->GetCore(); // shared_ptr that holds the Core while the lambda below (which captures that by val) works
    auto LoadNetworkAsync =
        [this, core, modelPath, network](const std::string& device) -> SoExecutableNetworkInternal {
            SoExecutableNetworkInternal executableNetwork;
            // std::cout << "!!! DEBUG: Starting Async loading to the " << device << " !!!" << std::endl;
            if (!modelPath.empty()) {
                executableNetwork = core->LoadNetwork(modelPath, device, {});
            } else {
                executableNetwork = core->LoadNetwork(network, device, {});
            }
            std::cout << "!!! DEBUG: " << device << " was loaded !!!" << std::endl;

            GenerateWorkers(device, executableNetwork);

            if (device.find("CPU") == std::string::npos) {
                _networkActualNeeded = executableNetwork;
                SetActualNetworkReadyStatus();
            }
            return executableNetwork;
    };

    // start CPU task
    const auto CPUIter = std::find_if(metaDevices.begin(), metaDevices.end(),
                                      [=](const DeviceInformation& d)->bool{return d.deviceName.find("CPU") != std::string::npos;});
    if (CPUIter != metaDevices.end()) {
        // to align with original MULTI
        _devicePriorities.push_back(*CPUIter);
        _config.insert(CPUIter->config.begin(), CPUIter->config.end());
        _cpuFuture = std::async(std::launch::async, LoadNetworkAsync, CPUIter->deviceName);
    }

    // start accelerator task, like GPU
    auto networkPrecision = GetNetworkPrecision(network);
    auto acceleratorDevice = _multiPlugin->SelectDevice(metaDevices, networkPrecision);
    bool isAccelerator =
        acceleratorDevice.deviceName.find("CPU") == std::string::npos;
    if (isAccelerator) {
        // to align with original MULTI
        _devicePriorities.push_back(acceleratorDevice);
        _config.insert(acceleratorDevice.config.begin(), acceleratorDevice.config.end());
        _acceleratorFuture = std::async(std::launch::async, LoadNetworkAsync, acceleratorDevice.deviceName);
    }

    // both are valid, like AUTO:CPU,GPU
    if (_cpuFuture.valid() && _acceleratorFuture.valid()) {
        try {
            _networkFirstReady = _cpuFuture.get();
            _alreadyActualNetwork = false;
        } catch (const std::exception& e) {
            printf("Warning: load network to CPU failed: %s\n", e.what());
            _networkActualNeeded = _acceleratorFuture.get();
            SetActualNetworkReadyStatus();
        }
    } else if (_acceleratorFuture.valid()) {  // only accelerator is valid, like AUTO:GPU
        _networkActualNeeded = _acceleratorFuture.get();
        SetActualNetworkReadyStatus();
    } else if (_cpuFuture.valid()) {  // only CPU is valid, like AUTO:CPU
        _networkActualNeeded = _cpuFuture.get();
        SetActualNetworkReadyStatus();
    } else {
        IE_THROW() << "No device task available";
    }
}

void MultiDeviceExecutableNetwork::SetActualNetworkReadyStatus() {
    _alreadyActualNetwork = true;
    try {
        _needPerfCounters = _networkActualNeeded->GetMetric(PluginConfigParams::KEY_PERF_COUNT).as<std::string>() == PluginConfigParams::YES;
    } catch (...) {
    }
}

void MultiDeviceExecutableNetwork::GenerateWorkers(const std::string& device, const SoExecutableNetworkInternal& executableNetwork) {
    auto itNumRequests = std::find_if(_devicePriorities.cbegin(), _devicePriorities.cend(),
                                      [&device](const DeviceInformation& d){ return d.deviceName == device;});
    unsigned int optimalNum = 0;
    try {
        optimalNum = executableNetwork->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    } catch (const InferenceEngine::Exception &iie) {
        IE_THROW()
            << "Every device used with the Multi-Device should "
            << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
            << "Failed to query the metric for the " << device << " with error:" << iie.what();
    }
    const auto numRequests = (_devicePriorities.end() == itNumRequests ||
                              itNumRequests->numRequestsPerDevices == -1) ? optimalNum : itNumRequests->numRequestsPerDevices;
    auto& workerRequests = _workerRequests[device];
    auto& idleWorkerRequests = _idleWorkerRequests[device];
    workerRequests.resize(numRequests);
    _inferPipelineTasksDeviceSpecific[device] = std::unique_ptr<ThreadSafeQueue<Task>>(new ThreadSafeQueue<Task>);
    auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
    idleWorkerRequests.set_capacity(numRequests);
    for (auto&& workerRequest : workerRequests) {
        workerRequest._inferRequest = { executableNetwork, executableNetwork->CreateInferRequest() };
        auto* workerRequestPtr = &workerRequest;
        IE_ASSERT(idleWorkerRequests.try_push(workerRequestPtr) == true);
        workerRequest._inferRequest->SetCallback(
            [workerRequestPtr, this, device, idleWorkerRequestsPtr] (std::exception_ptr exceptionPtr) mutable {
                IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                workerRequestPtr->_exceptionPtr = exceptionPtr;
                {
                    auto capturedTask = std::move(workerRequestPtr->_task);
                    capturedTask();
                }
                // try to return the request to the idle list (fails if the overall object destruction has began)
                if (idleGuard.Release()->try_push(workerRequestPtr)) {
                    // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                    // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                    Task t;
                    if (_inferPipelineTasks.try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t));
                    else if (_inferPipelineTasksDeviceSpecific[device]->try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t), device);
                }
            });
    }
}

void MultiDeviceExecutableNetwork::ScheduleToWorkerInferRequest(Task inferPipelineTask, DeviceName preferred_device) {
    auto devices = [&] {
        std::lock_guard<std::mutex> lock(_mutex);
        return _devicePriorities;
    }();
    for (auto&& device : devices) {
        if (!preferred_device.empty() && (device.deviceName != preferred_device))
            continue;
        WorkerInferRequest* workerRequestPtr = nullptr;
        NotBusyWorkerRequests& idleWorkerRequests = _idleWorkerRequests[device.deviceName];
        if (idleWorkerRequests.try_pop(workerRequestPtr)) {
            IdleGuard idleGuard{workerRequestPtr, idleWorkerRequests};
            _thisWorkerInferRequest = workerRequestPtr;
            {
                auto capturedTask = std::move(inferPipelineTask);
                capturedTask();
            }
            idleGuard.Release();
            return;
        }
    }
    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty())
        _inferPipelineTasksDeviceSpecific[preferred_device]->push(std::move(inferPipelineTask));
    else
        _inferPipelineTasks.push(std::move(inferPipelineTask));
}

void MultiDeviceExecutableNetwork::run(Task inferPipelineTask) {
    ScheduleToWorkerInferRequest(std::move(inferPipelineTask), _thisPreferredDeviceName);
}

MultiDeviceExecutableNetwork::~MultiDeviceExecutableNetwork() {
    // this is necessary to guarantee member destroyed after getting future
    if (!_alreadyActualNetwork) {
        // printf("!!! DEBUG: actual network is still not ready, wait that\n");
        _acceleratorFuture.get();
    }

    {
        std::lock_guard<std::mutex> lock(_mutex);
        _devicePriorities.clear();
    }
    /* NOTE: The only threads that use `MultiDeviceExecutableNetwork` worker infer requests' threads.
     *       But AsyncInferRequest destructor should wait for all asynchronous tasks by the request
     */
    for (auto&& idleWorker : _idleWorkerRequests) {
        // stop accepting any idle requests back (for re-scheduling)
        idleWorker.second.set_capacity(0);
    }
    _workerRequests.clear();
}

bool MultiDeviceExecutableNetwork::TryGetActualNetwork(InferenceEngine::SoExecutableNetworkInternal& soExecNetwork) {
    // if already get actual network
    if (_alreadyActualNetwork) {
        soExecNetwork = _networkActualNeeded;
        // reapply config to actual network
        // fixme: GPU doesn't support SetConfig and throw exception
        try {
            _networkActualNeeded->SetConfig(_cacheConfig);
        } catch (...) {
        }
        return true;
    }
    return false;
}

void MultiDeviceExecutableNetwork::WaitForActualDevice() const {
    if (_alreadyActualNetwork) {
        return;
    }

    if (_acceleratorFuture.valid()) {
        _networkActualNeeded = _acceleratorFuture.get();
        _alreadyActualNetwork = true;
    } else {
        IE_THROW() << "Export failed due to no valid executable network";
    }
}

RemoteContext::Ptr MultiDeviceExecutableNetwork::GetContext() const {
    if (_workModeIsAUTO) {
        WaitForActualDevice();
        return _networkActualNeeded->GetContext();
    }

    auto devices = [&] {
        std::lock_guard<std::mutex> lock(_mutex);
        return _devicePriorities;
    }();

    std::string devices_names;
    for (auto&& device : devices) {
        devices_names += device.deviceName + " ";
        const auto& n  = _networksPerDevice.at(device.deviceName);
        try {
            return n->GetContext();
        } catch (const NotImplemented&) {}
    }
    IE_THROW(NotImplemented) << "None of the devices in the MULTI has an associated remote context."
                       << " Current list of devices allowed via the DEVICE_PRIORITIES config: " << devices_names;
}

InferenceEngine::IInferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                InferenceEngine::OutputsDataMap networkOutputs) {
    auto num = _numRequestsCreated++;
    size_t sum = 0;
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    // borrowing device-specific blobs from the underlying requests for the device-agnostic, user-facing requests
    // this allows to potentially save on the data-copy later (if the requests are scheduled in the same order)
    for (const auto& device : _devicePrioritiesInitial) {
        auto& dev_requests = _workerRequests[device.deviceName];
        if ((num - sum) < dev_requests.size()) {
            request_to_share_blobs_with = dev_requests.at(num - sum)._inferRequest;
            break;
        }
        sum += dev_requests.size();
    }
    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
}

IInferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequest() {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    return std::make_shared<MultiDeviceAsyncInferRequest>(std::static_pointer_cast<MultiDeviceInferRequest>(syncRequestImpl),
                                                          _needPerfCounters,
                                                          std::static_pointer_cast<MultiDeviceExecutableNetwork>(shared_from_this()),
                                                          _callbackExecutor);
}

void MultiDeviceExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) {
    auto priorities = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == config.end() || config.size() > 1) {
        IE_THROW() << "The only config supported for the Network's SetConfig is MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES";
    } else {
        auto multiPlugin = std::dynamic_pointer_cast<MultiDeviceInferencePlugin>(this->_plugin);
        assert(multiPlugin != nullptr);
        auto metaDevices = multiPlugin->ParseMetaDevices(priorities->second, {});

        if (std::any_of(metaDevices.begin(), metaDevices.end(), [](const DeviceInformation& kvp) {
                return kvp.numRequestsPerDevices != -1;
            })) {
            IE_THROW() << "You can only change device priorities but not number of requests"
                     <<" with the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES!";
        }

        {
            std::lock_guard<std::mutex> lock{_mutex};
            for (auto && device : metaDevices) {
                if (_networksPerDevice.find(device.deviceName) == _networksPerDevice.end()) {
                    IE_THROW(NotFound) << "You can only change device priorities but not add new devices with"
                        << " the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES. "
                        << device.deviceName <<
                            " device was not in the original device list!";
                }
            }
            _devicePriorities = metaDevices;

            // update value in config
            _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = priorities->second;
        }
    }
}

InferenceEngine::Parameter MultiDeviceExecutableNetwork::GetConfig(const std::string &name) const {
    auto it = _config.find(name);
    if (it != _config.end()) {
        return it->second;
    } else {
        IE_THROW(NotFound) << name <<" not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter MultiDeviceExecutableNetwork::GetMetric(const std::string &name) const {
    if (_workModeIsAUTO) {
        // fixme: should we wait actual device? meanwhile it will block inference, how to fix?
        if (_alreadyActualNetwork) {
            return _networkActualNeeded->GetMetric(name);
        }
        return _networkFirstReady->GetMetric(name);
    }

    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int res = 0u;
        for (auto n : _networksPerDevice) {
            try {
                res += n.second->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const InferenceEngine::Exception &iie) {
                  IE_THROW()
                        << "Every device used with the Multi-Device should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << n.first << " with error:" << iie.what();
           }
        }
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, res);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        auto it = _networksPerDevice.begin();
        IE_ASSERT(it != _networksPerDevice.end());
        IE_SET_METRIC_RETURN(NETWORK_NAME, it->second->GetMetric(
            METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = { MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        IE_THROW() << "Unsupported Network metric: " << name;
    }
}

}  // namespace MultiDevicePlugin
