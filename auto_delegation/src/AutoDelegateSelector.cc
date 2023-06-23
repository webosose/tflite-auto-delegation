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
            auto idx = plan[i];
            auto registration = nodes[idx].second;

            auto op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
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

        if (apm.getPolicy() != AccelerationPolicyManager::kCPUOnly)
            return setTfLiteGPUDelegate(interpreter, apm);
        else
            return true;
    }

#ifdef USE_NPU
    bool AutoDelegateSelector::setWebOSNPUDelegate(tflite::Interpreter &interpreter)
    {
        webos::npu::tflite::NpuDelegateOptions npu_opts = webos::npu::tflite::NpuDelegateOptions();
        auto delegatePtr = TfLiteDelegatePtr(
                webos::npu::tflite::TfLiteNpuDelegateCreate(npu_opts),
                webos::npu::tflite::TfLiteNpuDelegateDelete);
        if (interpreter.ModifyGraphWithDelegate(delegatePtr.get()) != kTfLiteOk)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting webOS NPU delegate");
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

    bool AutoDelegateSelector::setTfLiteGPUDelegate(tflite::Interpreter &interpreter, AccelerationPolicyManager &apm)
    {
        auto policy = apm.getPolicy();
        TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        if (policy == AccelerationPolicyManager::kMinimumLatency)
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
            gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
            gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }
        else if (policy == AccelerationPolicyManager::kPytorchModelGPU)
        {
            gpu_opts.cpu_fallback_percentage = apm.getCPUFallbackPercentage();
            gpu_opts.is_pytorch_converted_model = true;
            gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
            gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }

#ifdef GPU_DELEGATE_ONLY_GL
        gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
#else
        gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
#endif

        auto *delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
        if (interpreter.ModifyGraphWithDelegate(delegate) != kTfLiteOk)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting TfLiteGPU delegate");
            return false;
        }

        return true;
    }

} // end of namespace aif
