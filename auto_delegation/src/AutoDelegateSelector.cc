/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include "AutoDelegateSelector.h"
#include "tools/Utils.h"

namespace
{
    static PmLogContext s_pmlogCtx = aif::getADPmLogContext();
} // end of anonymous namespace

namespace aif
{
    AutoDelegateSelector::AutoDelegateSelector(tflite::ops::builtin::BuiltinOpResolver *resolver)
        : resolver_(resolver)
    {
        // add custom webOSNPU operation to resolver
#ifdef USE_EDGETPU
        // add custom EdgeTPU operation to resolver
        resolver_->AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
#endif
    }

    AutoDelegateSelector::~AutoDelegateSelector()
    {
#ifdef USE_EDGETPU
        edgetpuContext_.reset();
#endif
    }

    bool AutoDelegateSelector::SelectDelegate(std::unique_ptr<tflite::Interpreter> *interpreter, AccelerationPolicyManager *apm)
    {

        if ((apm->GetPolicy() != AccelerationPolicyManager::kEnableLoadBalancing || apm->GetPolicy() != AccelerationPolicyManager::kPytorchModelGPU) && apm->GetCPUFallbackPercentage() != 0)
        {
            PmLogInfo(s_pmlogCtx, "ADS", 0, "current policy does not use load balancing but a non-zero value"
                                            "is set to cpu fallback percentage. So, the value set for cpu fallback percentage is ignored.");
        }

        tflite::Subgraph &subgraph = (*interpreter)->primary_subgraph();

        auto plan = subgraph.execution_plan();
        auto nodes = subgraph.nodes_and_registration();
        for (int i = 0; i < plan.size(); i++)
        {
            auto idx = plan[i];
            auto registration = nodes[idx].second;

            auto op = static_cast<tflite::BuiltinOperator>(registration.builtin_code);
            if (registration.custom_name != nullptr)
            {
                // check if the model is NPU compiled
                if (strcmp(registration.custom_name, "webosnpu-custom-op") == 0)
                {
                    if (SetWebOSNPUDelegate(interpreter) == true)
                    {
                        break;
                    }
                    else
                    {
                        PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting webOSNPU delegate");
                        return false;
                    }
                }
#ifdef USE_EDGETPU
                else if (strcmp(registration.custom_name, "edgetpu-custom-op") == 0)
                {
                    if (SetEdgeTPUDelegate(interpreter) == true)
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

        if (apm->GetPolicy() != AccelerationPolicyManager::kCPUOnly)
            return SetTfLiteGPUDelegate(interpreter, apm);
        else
            return true;
    }

    bool AutoDelegateSelector::SetWebOSNPUDelegate(std::unique_ptr<tflite::Interpreter> *interpreter)
    {
        return true;
    }

#ifdef USE_EDGETPU
    bool AutoDelegateSelector::SetEdgeTPUDelegate(std::unique_ptr<tflite::Interpreter> *interpreter)
    {
        edgetpuContext_ = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
        if (edgetpuContext_ == nullptr)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Cannot get EdgeTPU context");
            return false;
        }

        (*interpreter)->SetExternalContext(kTfLiteEdgeTpuContext, edgetpuContext_.get());
        PmLogInfo(s_pmlogCtx, "ADS", 0, "EdgeTPU Delegate is selected.");
        return true;
    }
#endif

    bool AutoDelegateSelector::SetTfLiteGPUDelegate(std::unique_ptr<tflite::Interpreter> *interpreter, AccelerationPolicyManager *apm)
    {
        auto policy = apm->GetPolicy();
        TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        if (policy == AccelerationPolicyManager::kMinimumLatency)
        {
            gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }
        else if (policy == AccelerationPolicyManager::kEnableLoadBalancing)
        {
            gpu_opts.cpu_fallback_percentage = apm->GetCPUFallbackPercentage();
            gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
            gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }
        else if (policy == AccelerationPolicyManager::kPytorchModelGPU)
        {
            gpu_opts.cpu_fallback_percentage = apm->GetCPUFallbackPercentage();
            gpu_opts.is_pytorch_converted_model = true;
            gpu_opts.inference_priority1 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
            gpu_opts.inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
            gpu_opts.inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        }

        auto *delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
        if ((*interpreter)->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
        {
            PmLogError(s_pmlogCtx, "ADS", 0, "Something went wrong while setting TfLiteGPU delegate");
            return false;
        }

        return true;
    }

} // end of namespace aif
