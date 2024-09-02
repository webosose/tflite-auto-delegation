/*
 * Copyright (c) 2022 LG Electronics Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "AccelerationPolicyManager.h"
#include "tools/Logger.h"

namespace
{
    static PmLogContext s_pmlogCtx = aif::getADPmLogContext();
} // end of anonymous namespace

namespace aif
{
    AccelerationPolicyManager::AccelerationPolicyManager()
    {
    }

    AccelerationPolicyManager::AccelerationPolicyManager(const std::string &config)
    {
        rapidjson::Document d;
        d.Parse(config.c_str());

        if (!d.HasParseError() && d.FindMember("policy") != d.MemberEnd())
        {
            Policy policy = stringToPolicy(d["policy"].GetString());
            setPolicy(policy);

            if (d.FindMember("cpu_fallback_percentage") != d.MemberEnd())
            {
                int cpuFallbackPercentage = d["cpu_fallback_percentage"].GetInt();
                setCPUFallbackPercentage(cpuFallbackPercentage);
            }
        }
        if (!d.HasParseError() && d.HasMember("serialization"))
        {
            if (d["serialization"].HasMember("dir_path") && d["serialization"].HasMember("model_token"))
            {
                Caching cache;

                cache.useCache = true;
                cache.serialization_dir = d["serialization"]["dir_path"].IsString() ? d["serialization"]["dir_path"].GetString() : "";
                cache.model_token = d["serialization"]["model_token"].IsString() ? d["serialization"]["model_token"].GetString() : "";
                setCache(std::move(cache));
            }
            else
            {
                PmLogError(s_pmlogCtx, "APM", 0, "dir_path or model_token is invalid");
            }
        }
    }

    AccelerationPolicyManager::~AccelerationPolicyManager()
    {
    }

    bool AccelerationPolicyManager::setPolicy(AccelerationPolicyManager::Policy policy)
    {
        m_policy = policy;

        switch (m_policy)
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

    AccelerationPolicyManager::Policy AccelerationPolicyManager::getPolicy()
    {
        return m_policy;
    }

    bool AccelerationPolicyManager::setCPUFallbackPercentage(int percentage)
    {
        if (percentage < 0)
            percentage = 0;
        else if (percentage > 100)
            percentage = 100;

        m_cpuFallbackPercentage = percentage;

        PmLogInfo(s_pmlogCtx, "APM", 0, "Set CPU Fallback Percentage: %d", m_cpuFallbackPercentage);

        return true;
    }

    int AccelerationPolicyManager::getCPUFallbackPercentage()
    {
        return m_cpuFallbackPercentage;
    }

    AccelerationPolicyManager::Policy AccelerationPolicyManager::stringToPolicy(const std::string &policyStr)
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

    void AccelerationPolicyManager::setCache(AccelerationPolicyManager::Caching cache)
    {
        m_cache = std::move(cache);
    }

    AccelerationPolicyManager::Caching AccelerationPolicyManager::getCache()
    {
        return m_cache;
    }
} // end of namespace aif
