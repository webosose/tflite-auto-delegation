/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: LicenseRef-LGE-Proprietary
 */
#include "AccelerationPolicyManager.h"

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
    PmLogContext ad_context = nullptr;
    PmLogGetContext("auto_delegation", &ad_context);

    policy_ = policy;

    switch (policy_)
    {
    case kCPUOnly:
        PmLogInfo(ad_context, "APM", 0, "Set Acceleration Policy: CPU Only");
        break;
    case kMaximumPrecision:
        PmLogInfo(ad_context, "APM", 0, "Set Acceleration Policy: Maximum Precision");
        break;
    case kMinimumLatency:
        PmLogInfo(ad_context, "APM", 0, "Set Acceleration Policy: Minimum Latency");
        break;
    case kEnableLoadBalancing:
        PmLogInfo(ad_context, "APM", 0, "Set Acceleration Policy: Enable Load Balancing");
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
    PmLogContext ad_context = nullptr;
    PmLogGetContext("auto_delegation", &ad_context);

    if (percentage < 0)
        percentage = 0;
    else if (percentage > 100)
        percentage = 100;

    cpu_fallback_percentage_ = percentage;

    PmLogInfo(ad_context, "APM", 0, "Set CPU Fallback Percentage: %d", cpu_fallback_percentage_);

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

    return policy;
}