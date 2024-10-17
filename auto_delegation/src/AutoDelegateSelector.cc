/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "AutoDelegateSelector.h"
#include "tools/Logger.h"

namespace
{
    static PmLogContext s_pmlogCtx = aif::getADPmLogContext();
} // end of anonymous namespace

namespace aif
{
    AutoDelegateSelector::AutoDelegateSelector()
    {
    }

    bool AutoDelegateSelector::selectDelegate(tflite::Interpreter &interpreter, AccelerationPolicyManager &apm)
    {
        if (apm.getCPUFallbackPercentage() != 0)
        {
            if (apm.getPolicy() != AccelerationPolicyManager::kEnableLoadBalancing &&
                apm.getPolicy() != AccelerationPolicyManager::kPytorchModelGPU)
            {
                PmLogInfo(s_pmlogCtx, "ADS", 0, "current policy does not use load balancing but a non-zero value"
                                                "is set to cpu fallback percentage. So, the value set for cpu fallback percentage is ignored.");
            }
        }

        tflite::Subgraph &subgraph = interpreter.primary_subgraph();

        auto plan = subgraph.execution_plan();
        auto nodes = subgraph.nodes_and_registration();
        for (int i = 0; i < plan.size(); i++)
        {
            int idx = plan[i];
            if (idx < 0 || idx >= nodes.size())
            {
                PmLogError(s_pmlogCtx, "ADS", 0, "Execution plan index error");
                return false;
            }
            const auto &registration = nodes[idx].second;

            if (registration.custom_name != nullptr)
            {
#ifdef USE_NPU
                // check if the model is NPU compiled
                if (strcmp(registration.custom_name, "lgnpu_custom_op") == 0)
                {
                    if (setWebOSNPUDelegate(interpreter) == true)
                    {
                        break;
                    }
                    else
                    {
                        PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting webOSNPU delegate");
                        return false;
                    }
                }
#endif
#ifdef USE_EDGETPU
                if (strcmp(registration.custom_name, "edgetpu-custom-op") == 0)
                {
                    if (setEdgeTPUDelegate(interpreter) == true)
                    {
                        break;
                    }
                    else
                    {
                        PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting EdgeTPU delegate");
                        return false;
                    }
                }
#endif
            }
        }
#ifdef USE_NNAPI
        if (apm.getPolicy() == AccelerationPolicyManager::kMinRes) {
            return setNNAPIDelegate(interpreter, apm);
        }
        else if(apm.getPolicy() == AccelerationPolicyManager::kMinLatencyMinRes) {
            if (!setNNAPIDelegate(interpreter, apm)) {
                PmLogError(s_pmlogCtx, "ADS", 0, "Fail to get Policy while using NNAPI");
                return false;
            }
        }
#endif
#ifdef USE_GPU
        if (apm.getPolicy() != AccelerationPolicyManager::kCPUOnly)
            return setTfLiteGPUDelegate(interpreter, apm);
#endif
        return true;
    }


#ifdef USE_GPU
    bool AutoDelegateSelector::setTfLiteGPUDelegate(tflite::Interpreter &interpreter, AccelerationPolicyManager &apm)
    {
        bool isIMG = false;
#ifdef GPU_DELEGATE_ONLY_CL
        isIMG = isCLDeviceVendorIMG();
#endif
        auto policy = apm.getPolicy();
        TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        if (policy & AccelerationPolicyManager::kMinimumLatency)
        {
            gpu_opts.cpu_fallback_percentage = 0;
            gpu_opts.is_pytorch_converted_model = false;
            gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }
        else if (policy == AccelerationPolicyManager::kEnableLoadBalancing)
        {
            gpu_opts.cpu_fallback_percentage = apm.getCPUFallbackPercentage();
            gpu_opts.is_pytorch_converted_model = false;
            if (isIMG)
            {
                gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            }
            else
            {
                gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
            }
            gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }
        else if (policy == AccelerationPolicyManager::kPytorchModelGPU)
        {
            gpu_opts.cpu_fallback_percentage = apm.getCPUFallbackPercentage();
            gpu_opts.is_pytorch_converted_model = true;
            if (isIMG)
            {
                gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            }
            else
            {
                gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
            }
            gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }

        const auto &cache = apm.getCache();
        if (cache.useCache)
        {
            gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
            gpu_opts.serialization_dir = cache.serialization_dir.c_str();
            gpu_opts.model_token = cache.model_token.c_str();
        }

#ifdef GPU_DELEGATE_ONLY_GL
        gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
#elif GPU_DELEGATE_ONLY_CL
        gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
#endif

        auto deleter = [](TfLiteDelegate* delegate) {
                TfLiteGpuDelegateV2Delete(delegate);
        };

        TfLiteDelegate* raw_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
        std::unique_ptr<TfLiteDelegate, decltype(deleter)> delegate(raw_delegate, deleter);
        if (interpreter.ModifyGraphWithDelegate(std::move(delegate)) != kTfLiteOk)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting TfLiteGPU delegate");
            return false;
        }

        return true;
    }

#ifdef GPU_DELEGATE_ONLY_CL
    bool AutoDelegateSelector::isCLDeviceVendorIMG()
    {
        cl_uint platformCount;
        if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Error: clGetPlatformIDs failed.");
            return false;
        }

        std::vector<cl_platform_id> platforms(platformCount);
        if (clGetPlatformIDs(platformCount, platforms.data(), nullptr) != CL_SUCCESS)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Error: clGetPlatformIDs failed.");
            return false;
        }

        if (platformCount == 0)
        {
            PmLogWarning(s_pmlogCtx, "ADS", 0, "Warning: Expected exactly one OpenCL platform, but found no platform.");
            return false;
        }
        else if (platformCount != 1)
        {

            PmLogWarning(s_pmlogCtx, "ADS", 0, "Warning: Expected exactly one OpenCL platform, but found %d platforms.", platformCount);
        }

        cl_uint deviceCount;
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);

        if (deviceCount == 0)
        {
            PmLogWarning(s_pmlogCtx, "ADS", 0, "Warning: Expected exactly one OpenCL device, but found no device.");
            return false;
        }
        else if (deviceCount != 1)
        {

            PmLogWarning(s_pmlogCtx, "ADS", 0, "Warning: Expected exactly one OpenCL device, but found %d devices.", deviceCount);
        }

        char buffer[256];
        if (clGetDeviceInfo(devices[0], CL_DEVICE_VENDOR, sizeof(buffer), buffer, nullptr) != CL_SUCCESS)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Error: clGetDeviceInfo failed.");
            return false;
        }
        PmLogInfo(s_pmlogCtx, "ADS", 0, "CL Device Vendor: %s", buffer);

        const std::string img_vendor_name = "Imagination Technologies";

        return std::string(buffer) == img_vendor_name ? true : false;
    }
#endif
#endif

#ifdef USE_NPU
    bool AutoDelegateSelector::setWebOSNPUDelegate(tflite::Interpreter &interpreter)
    {
        webos::npu::tflite::NpuDelegateOptions npu_opts = webos::npu::tflite::NpuDelegateOptions();
        auto delegatePtr = tflite::Interpreter::TfLiteDelegatePtr(
            webos::npu::tflite::TfLiteNpuDelegateCreate(npu_opts),
            webos::npu::tflite::TfLiteNpuDelegateDelete);
        if (interpreter.ModifyGraphWithDelegate(std::move(delegatePtr)) != kTfLiteOk)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting webOS NPU delegate");
            return false;
        }
        return true;
    }
#endif

#ifdef USE_NNAPI
    bool AutoDelegateSelector::setNNAPIDelegate(tflite::Interpreter &interpreter, AccelerationPolicyManager &apm)
    {
        auto policy = apm.getPolicy();
        tflite::StatefulNnApiDelegate::Options nnapi_opts = tflite::StatefulNnApiDelegate::Options();
        const auto& cache = apm.getNnapiCache();

        if(policy == AccelerationPolicyManager::kMinRes || policy == AccelerationPolicyManager::kMinLatencyMinRes) {
            if (cache.cache_dir != "" && cache.model_token != "")
            {
                nnapi_opts.cache_dir   = cache.cache_dir.c_str();
                nnapi_opts.model_token = cache.model_token.c_str();
            }
            else{
                PmLogError(s_pmlogCtx, "ADS", 0, "cache_dir or model_token is invalid");
            }
            if (cache.disallow_nnapi_cpu)
            {
                nnapi_opts.disallow_nnapi_cpu = cache.disallow_nnapi_cpu;
            }
            else{
                PmLogError(s_pmlogCtx, "ADS", 0, "disallow_nnapi_cpu is invalid");
            }
            if (cache.max_number_delegated_partitions != 0)
            {
                nnapi_opts.max_number_delegated_partitions = cache.max_number_delegated_partitions;
            }
            else{
                PmLogError(s_pmlogCtx, "ADS", 0, "max_number_delegated_partitions is invalid");
            }
            if (cache.accelerator_name != "")
            {
                nnapi_opts.accelerator_name = cache.accelerator_name.c_str();
            }
            else{
                PmLogError(s_pmlogCtx, "ADS", 0, "accelerator_name is invalid");
            }
        }

        auto deleter = [](TfLiteDelegate* delegate) { delete delegate; };

        TfLiteDelegate* nnapi_delegate = new tflite::StatefulNnApiDelegate(nnapi_opts);
        std::unique_ptr<TfLiteDelegate, decltype(deleter)> delegatePtr(nnapi_delegate, deleter);
        if (interpreter.ModifyGraphWithDelegate(std::move(delegatePtr)) != kTfLiteOk)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting TfLite NNAPI delegate");
            return false;
        }
        return true;
    }
#endif

#ifdef USE_EDGETPU
    bool AutoDelegateSelector::setEdgeTPUDelegate(tflite::Interpreter &interpreter)
    {
        auto delegate_options = TfLiteExternalDelegateOptionsDefault(EDGETPU_LIB_PATH.c_str());

        auto external_delegate = TfLiteExternalDelegateCreate(&delegate_options);
        if (interpreter.ModifyGraphWithDelegate(external_delegate) != kTfLiteOk)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting TPU delegate");
            return false;
        }
        return true;
    }
#endif

} // end of namespace aif
