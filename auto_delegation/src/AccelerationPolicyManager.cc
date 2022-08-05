/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include "AccelerationPolicyManager.h"
#include "tools/Utils.h"

namespace
{

static PmLogContext s_pmlogCtx = aif::getADPmLogContext();

} // end of anonymous namespace

namespace aif {

AccelerationPolicyManager::AccelerationPolicyManager()
{
}

AccelerationPolicyManager::AccelerationPolicyManager(std::string config)
{
    rapidjson::Document d;
    d.Parse(config.c_str());

    if (!d.HasParseError() && d.FindMember("policy") != d.MemberEnd())
    {
        Policy policy = stringToPolicy(d["policy"].GetString());
        SetPolicy(policy);

        if (d.FindMember("cpu_fallback_percentage") != d.MemberEnd())
        {
            int cpuFallbackPercentage = d["cpu_fallback_percentage"].GetInt();
            SetCPUFallbackPercentage(cpuFallbackPercentage);
        }
    }
}

AccelerationPolicyManager::~AccelerationPolicyManager()
{
}

bool AccelerationPolicyManager::SetPolicy(AccelerationPolicyManager::Policy policy)
{
    policy_ = policy;

    switch (policy_)
    {
    case kCPUOnly:
        PmLogInfo(s_pmlogCtx, "APM", 0, "Set Acceleration Policy: CPU Only");
        break;
    case kMaximumPrecision:
        PmLogInfo(s_pmlogCtx, "APM", 0, "Set Acceleration Policy: Maximum Precision");
        break;
    case kMinimumLatency:
        PmLogInfo(s_pmlogCtx, "APM", 0, "Set Acceleration Policy: Minimum Latency");
        break;
    case kEnableLoadBalancing:
        PmLogInfo(s_pmlogCtx, "APM", 0, "Set Acceleration Policy: Enable Load Balancing");
        break;
    case kPytorchModelGPU:
        PmLogInfo(s_pmlogCtx, "APM", 0, "Set Acceleration Policy: PyTorch Model GPU");
        break;
    default:
        break;
    }

    return true;
}

AccelerationPolicyManager::Policy AccelerationPolicyManager::GetPolicy()
{
    return policy_;
}

bool AccelerationPolicyManager::SetCPUFallbackPercentage(int percentage)
{
    if (percentage < 0)
        percentage = 0;
    else if (percentage > 100)
        percentage = 100;

    cpu_fallback_percentage_ = percentage;

    PmLogInfo(s_pmlogCtx, "APM", 0, "Set CPU Fallback Percentage: %d", cpu_fallback_percentage_);

    return true;
}

int AccelerationPolicyManager::GetCPUFallbackPercentage()
{
    return cpu_fallback_percentage_;
}

AccelerationPolicyManager::Policy AccelerationPolicyManager::stringToPolicy(std::string policyStr)
{

    AccelerationPolicyManager::Policy policy = AccelerationPolicyManager::Policy::kCPUOnly;
    if (policyStr.compare("CPU_ONLY") == 0)
        policy = AccelerationPolicyManager::Policy::kCPUOnly;
    else if (policyStr.compare("MAX_PRECISION") == 0)
        policy = AccelerationPolicyManager::Policy::kMaximumPrecision;
    else if (policyStr.compare("MIN_LATENCY") == 0)
        policy = AccelerationPolicyManager::Policy::kMinimumLatency;
    else if (policyStr.compare("LOAD_BALANCING") == 0)
        policy = AccelerationPolicyManager::Policy::kEnableLoadBalancing;
    else if (policyStr.compare("PYTORCH_MODEL_GPU") == 0)
        policy = AccelerationPolicyManager::Policy::kPytorchModelGPU;

    return policy;
}

} // end of namespace aif
