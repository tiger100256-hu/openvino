// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_command_queues_builder.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include <string>

namespace cldnn {
namespace ocl {

command_queues_builder::command_queues_builder()
    : _profiling(false),
      _out_of_order(false),
      _supports_queue_families(false),
      _priority_mode(priority_mode_types::disabled),
      _throttle_mode(throttle_mode_types::disabled) {}

std::vector<cl_queue_properties> command_queues_builder::get_properties(const cl::Device& device, uint16_t stream_id) {
    std::vector<cl_queue_properties> properties;

    if (_priority_mode != priority_mode_types::disabled) {
        unsigned cl_queue_priority_value = CL_QUEUE_PRIORITY_MED_KHR;
        switch (_priority_mode) {
            case priority_mode_types::high:
                cl_queue_priority_value = CL_QUEUE_PRIORITY_HIGH_KHR;
                break;
            case priority_mode_types::low:
                cl_queue_priority_value = CL_QUEUE_PRIORITY_LOW_KHR;
                break;
            default:
                break;
        }

        properties.insert(properties.end(), {CL_QUEUE_PRIORITY_KHR, cl_queue_priority_value});
    }

    if (_throttle_mode != throttle_mode_types::disabled) {
        unsigned cl_queue_throttle_value = CL_QUEUE_THROTTLE_MED_KHR;
        switch (_throttle_mode) {
            case throttle_mode_types::high:
                cl_queue_throttle_value = CL_QUEUE_THROTTLE_HIGH_KHR;
                break;
            case throttle_mode_types::low:
                cl_queue_throttle_value = CL_QUEUE_THROTTLE_LOW_KHR;
                break;
            default:
                break;
        }

        properties.insert(properties.end(), {CL_QUEUE_THROTTLE_KHR, cl_queue_throttle_value});
    }

    if (_supports_queue_families) {
        cl_uint num_queues = 0;
        cl_uint family = 0;

        std::vector<cl_queue_family_properties_intel> qfprops = device.getInfo<CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL>();
        for (cl_uint q = 0; q < qfprops.size(); q++) {
            if (qfprops[q].capabilities == CL_QUEUE_DEFAULT_CAPABILITIES_INTEL && qfprops[q].count > num_queues) {
                family = q;
                num_queues = qfprops[q].count;
            }
        }

        if (num_queues)
            properties.insert(properties.end(), {CL_QUEUE_FAMILY_INTEL, family,
                                                 CL_QUEUE_INDEX_INTEL, stream_id % num_queues});
    }

    cl_command_queue_properties cl_queue_properties =
        ((_profiling ? CL_QUEUE_PROFILING_ENABLE : 0) | (_out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0));

    properties.insert(properties.end(), {CL_QUEUE_PROPERTIES, cl_queue_properties, 0});

    return properties;
}

ocl_queue_type command_queues_builder::build(const cl::Context& context, const cl::Device& device) {
    ocl_queue_type queue;
    cl_int error_code = CL_SUCCESS;
    static std::atomic<uint16_t> stream_id{0};

    auto properties = get_properties(device, stream_id++);

    queue = clCreateCommandQueueWithProperties(context.get(), device.get(), properties.data(), &error_code);

    if (error_code != CL_SUCCESS) {
        CLDNN_ERROR_MESSAGE("Command queues builders",
                            "clCreateCommandQueueWithPropertiesINTEL error " + std::to_string(error_code));
    }

    return queue;
}

void command_queues_builder::set_priority_mode(priority_mode_types priority, bool extension_support) {
    if (priority != priority_mode_types::disabled && !extension_support) {
        CLDNN_ERROR_MESSAGE("Command queues builders - priority_mode",
                            std::string("The param priority_mode is set in engine_configuration, ")
                            .append("but cl_khr_priority_hints or cl_khr_create_command_queue ")
                            .append("is not supported by current OpenCL implementation."));
    }
    _priority_mode = priority;
}

void command_queues_builder::set_throttle_mode(throttle_mode_types throttle, bool extension_support) {
    if (throttle != throttle_mode_types::disabled && !extension_support) {
        CLDNN_ERROR_MESSAGE("Command queues builders - throttle_mode",
                            std::string("The param throttle_mode is set in engine_configuration, ")
                            .append("but cl_khr_throttle_hints is not supported by current OpenCL implementation."));
    }
    _throttle_mode = throttle;
}

void command_queues_builder::set_supports_queue_families(bool extension_support) {
    _supports_queue_families = extension_support;
}
}  // namespace ocl
}  // namespace cldnn
